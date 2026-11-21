"""Resample images and labels onto the pixel grid of a coordinate system, so that they are tiled by plain slicing."""

from __future__ import annotations

import math
import shutil
import uuid
import warnings
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import dask
import dask.array as da
import numpy as np
from dask.array.core import normalize_chunks
from spatialdata import get_pyramid_levels, rasterize, read_zarr
from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, get_channel_names, get_model
from spatialdata.transformations import Identity, Sequence, get_transformation
from xarray import DataArray, DataTree

from .utils import element_to_cs_matrix, is_identity_xy

if TYPE_CHECKING:
    from collections.abc import Sequence as SequenceT
    from os import PathLike

    from spatialdata import SpatialData
    from spatialdata.transformations import BaseTransformation

__all__ = ["align_to_cs"]


@dataclass(frozen=True)
class _AlignmentSpec:
    """The aligned element that :func:`align_to_cs` writes for one source element, decided before anything is written."""

    source: str
    target: str
    model: type[Image2DModel] | type[Labels2DModel]
    height: int
    width: int
    scale_factors: list[int] | None


def _raster(sdata: SpatialData, name: str) -> DataArray | DataTree:
    """The element `name` of `sdata`, typed as a raster (:func:`_alignment_spec` checks that it is one)."""
    return cast("DataArray | DataTree", sdata[name])


def _spatial_shape(element: DataArray | DataTree) -> tuple[int, int]:
    """Height and width of the full-resolution level."""
    level = cast("DataArray", get_pyramid_levels(element, n=0)) if isinstance(element, DataTree) else element
    return int(level.sizes["y"]), int(level.sizes["x"])


def _pyramid_scale_factors(element: DataArray | DataTree, name: str) -> list[int] | None:
    """The relative downscaling factors between consecutive levels of `element`, as ``parse(scale_factors=...)`` takes.

    ``None`` for a single-scale element.
    """
    if not isinstance(element, DataTree):
        return None
    levels = cast("list[DataArray]", get_pyramid_levels(element))
    shapes = [(int(lv.sizes["y"]), int(lv.sizes["x"])) for lv in levels]
    factors: list[int] = []
    for (py, px), (cy, cx) in zip(shapes[:-1], shapes[1:], strict=True):
        fy, fx = round(py / cy), round(px / cx)
        if fy != fx or fy < 2 or py // fy != cy or px // fx != cx:
            raise ValueError(
                f"The pyramid of `{name}` (level shapes {shapes}) does not have one integer factor per level; pass "
                "`scale_factors` explicitly."
            )
        factors.append(fy)
    return factors or None


def _transformations(element: DataArray | DataTree, cs: str) -> dict[str, BaseTransformation]:
    """Transformations of the aligned element: the identity to `cs`, and the original placement in every other system.

    The aligned pixel grid *is* `cs`, so the original transformation to another system ``X`` becomes
    ``T_X ∘ T_cs⁻¹`` (first undo the transformation to `cs`, then apply the one to ``X``).
    """
    old = cast("dict[str, BaseTransformation]", get_transformation(element, get_all=True))
    to_cs = old[cs]
    out: dict[str, BaseTransformation] = {cs: Identity()}
    for system, transformation in old.items():
        if system != cs:
            out[system] = Sequence([to_cs.inverse(), transformation])
    return out


def _target_extent(sdata: SpatialData, name: str, cs: str, like: str | None) -> tuple[int, int]:
    """Height and width of the aligned array: those of `like`, or the bounding box of `name` in `cs` from the origin."""
    if like is not None:
        if not is_identity_xy(element_to_cs_matrix(_raster(sdata, like), cs)):
            raise ValueError(
                f"`like={like!r}` must be in the pixel grid of `{cs}` (its transformation is not the identity)."
            )
        return _spatial_shape(_raster(sdata, like))
    height, width = _spatial_shape(_raster(sdata, name))
    corners = np.array([[0, 0, 1], [width, 0, 1], [0, height, 1], [width, height, 1]], dtype=np.float64)
    mapped = element_to_cs_matrix(_raster(sdata, name), cs) @ corners.T
    max_x, max_y = float(mapped[0].max()), float(mapped[1].max())
    if max_x <= 0 or max_y <= 0:
        raise ValueError(f"`{name}` lies entirely at negative coordinates of `{cs}`, so nothing of it can be aligned.")
    return math.ceil(max_y - 1e-9), math.ceil(max_x - 1e-9)


def _alignment_spec(
    sdata: SpatialData,
    name: str,
    *,
    cs: str,
    target: str,
    like: str | None,
    scale_factors: SequenceT[int] | None,
) -> _AlignmentSpec | None:
    """Validate `name` and decide what to write for it, ``None`` if it is aligned already."""
    element = _raster(sdata, name)
    model = get_model(element)
    if model in (Image3DModel, Labels3DModel):
        raise NotImplementedError(f"Only 2D images and labels can be aligned, but `{name}` is 3D.")
    if model not in (Image2DModel, Labels2DModel):
        raise ValueError(f"`{name}` is not an image or labels element.")
    if cs not in cast("dict[str, Any]", get_transformation(element, get_all=True)):
        raise ValueError(f"`{name}` has no transformation to the coordinate system `{cs}`.")
    matrix = element_to_cs_matrix(element, cs)
    if is_identity_xy(matrix):
        warnings.warn(
            f"`{name}` is already in the pixel grid of `{cs}`; it is not resampled.", UserWarning, stacklevel=3
        )
        return None
    if abs(np.linalg.det(matrix[:2, :2])) < 1e-12:
        raise ValueError(f"The transformation of `{name}` to `{cs}` is not invertible.")
    height, width = _target_extent(sdata, name, cs, like)
    factors = list(scale_factors) if scale_factors is not None else _pyramid_scale_factors(element, name)
    raster_model = cast("type[Image2DModel] | type[Labels2DModel]", model)
    return _AlignmentSpec(name, target, raster_model, height, width, factors or None)


def _rasterize_block(
    element: DataArray | DataTree, cs: str, is_image: bool, dtype: np.dtype, block_info: dict[Any, Any] | None = None
) -> np.ndarray:
    """One block of the aligned array: :func:`spatialdata.rasterize` of `element` over the block, in pixels of `cs`.

    The block's bounds are integers, so its pixels are exactly those of a single rasterization of the whole extent.
    """
    assert block_info is not None
    (y0, y1), (x0, x1) = block_info[None]["array-location"][-2:]
    with dask.config.set(scheduler="synchronous"):  # a small computation nested in a task of the outer one
        tile = cast(
            "DataArray",
            rasterize(
                element,
                axes=("x", "y"),
                min_coordinate=[x0, y0],
                max_coordinate=[x1, y1],
                target_coordinate_system=cs,
                target_unit_to_pixels=1.0,
            ),
        )
        out = np.asarray(tile.values)
    if not is_image and out.ndim == 3:  # rasterize returns labels with a singleton channel axis
        out = out[0]
    if out.shape[-2:] != (y1 - y0, x1 - x0):
        raise RuntimeError(f"Resampling gave a block of shape {out.shape}, expected (..., {y1 - y0}, {x1 - x0}).")
    return out.astype(dtype, copy=False)


def _aligned_element(sdata: SpatialData, spec: _AlignmentSpec, cs: str, chunk: int) -> DataArray | DataTree:
    """The lazy, resampled element (computed from `sdata`, which is never the store being written).

    A single :func:`spatialdata.rasterize` of the whole extent is one dask block the size of the original image
    (``dask_image``'s default output chunking), too large for a whole-slide image. The array is therefore assembled
    from one rasterization per `chunk` x `chunk` block, which keeps memory at a few blocks.
    """
    element = _raster(sdata, spec.source)
    is_image = spec.model is Image2DModel
    level0 = cast("DataArray", get_pyramid_levels(element, n=0)) if isinstance(element, DataTree) else element
    dtype = level0.dtype
    spatial = (normalize_chunks(chunk, (spec.height,))[0], normalize_chunks(chunk, (spec.width,))[0])
    if is_image:
        channels = get_channel_names(element)
        chunks: tuple[tuple[int, ...], ...] = ((len(channels),), *spatial)
        dims: tuple[str, ...] = ("c", "y", "x")
    else:
        channels, chunks, dims = None, spatial, ("y", "x")
    # `partial`, not extra arguments: an xarray element is itself a dask collection, which map_blocks would try to
    # align block-wise with the output. An explicit name avoids hashing the element.
    data = da.map_blocks(
        partial(_rasterize_block, element, cs, is_image, dtype),
        chunks=chunks,
        dtype=dtype,
        meta=np.empty((0,) * len(dims), dtype=dtype),
        name=f"align-to-cs-{spec.source}-{uuid.uuid4().hex}",
    )
    # written with the chunk shape of the original (otherwise `parse` uses its small default, i.e. many more files)
    write_chunks = dict(zip(level0.dims, level0.data.chunksize, strict=True))
    return cast(
        "DataArray | DataTree",
        spec.model.parse(
            data,
            dims=dims,
            c_coords=channels,
            transformations=_transformations(element, cs),
            scale_factors=spec.scale_factors,
            chunks={str(d): int(write_chunks[d]) for d in dims},
        ),
    )


def _check_written(sdata: SpatialData, spec: _AlignmentSpec, cs: str) -> None:
    """The written element can be tiled by slicing: same checks as ``Loader.add_sdata`` and ``write_sharded_sdata``."""
    element = _raster(sdata, spec.target)
    if get_model(element) is not spec.model:
        raise RuntimeError(f"`{spec.target}` was written as {get_model(element).__name__}, not {spec.model.__name__}.")
    if not is_identity_xy(element_to_cs_matrix(element, cs)):
        raise RuntimeError(f"`{spec.target}` was written, but its transformation to `{cs}` is not the identity.")
    if _spatial_shape(element) != (spec.height, spec.width):
        raise RuntimeError(
            f"`{spec.target}` has shape {_spatial_shape(element)}, expected {(spec.height, spec.width)}."
        )


def align_to_cs(
    sdata_path: str | PathLike[str],
    elements: str | SequenceT[str],
    *,
    cs: str = "global",
    out_path: str | PathLike[str] | None = None,
    suffix: str = "_aligned",
    overwrite: bool = False,
    like: str | None = None,
    chunk: int = 4096,
    scale_factors: SequenceT[int] | None = None,
) -> SpatialData:
    """Resample images and labels onto the pixel grid of the coordinate system `cs`.

    An element whose transformation to `cs` is not the identity (for instance an H&E image registered to a Xenium
    slide, with a rotation, a scaling or a shear) cannot be tiled by slicing: :class:`spatialbatch.Loader` then makes
    one spatialdata call per tile, and :func:`spatialbatch.write_sharded_sdata` refuses it. This resamples it once with
    :func:`spatialdata.rasterize` so that pixel ``(0, 0)`` is the origin of `cs` and one pixel is one unit of `cs`.
    Afterwards its transformation to `cs` is the identity and it is tiled by slicing, in place or in a collection.

    Any affine transformation is supported. Resampling is nearest neighbor (as :func:`spatialdata.rasterize`), hence
    label ids are preserved. The transformations to the other coordinate systems are updated so that the element
    stays where it was in each of them.

    A tile of the aligned element is :func:`spatialdata.rasterize` of the original over the tile the loader reads,
    which starts at an integer pixel of `cs` as for every element already in the grid (e.g. the morphology image), so
    all modalities of a cell share one pixel grid. ``Loader(rasterize=True)`` on the original instead starts each tile
    at the fractional corner of the cell, so its tiles are shifted by up to half a pixel from these.

    Parameters
    ----------
    sdata_path
        A SpatialData zarr store.
    elements
        The images and/or labels to align. Elements already in the pixel grid of `cs` are skipped with a warning.
    cs
        The coordinate system whose pixel grid the elements are resampled onto.
    out_path
        If ``None``, the aligned elements are added to `sdata_path` as ``<name><suffix>``. Otherwise `sdata_path` is
        copied to `out_path` and each aligned element replaces the original under its own name there; `sdata_path`
        is not modified.
    suffix
        Suffix of the names of the aligned elements if `out_path` is ``None``.
    overwrite
        Whether to replace existing ``<name><suffix>`` elements, or an existing `out_path`.
    like
        An element in the pixel grid of `cs` whose height and width the aligned elements get (e.g. the morphology
        image), so that all modalities have the same extent. By default the extent is the bounding box of each
        element in `cs`, from the origin: what lies at negative coordinates is cropped.
    chunk
        Side in pixels of the blocks that are resampled independently, which bounds memory use (a few blocks per
        thread). The written arrays keep the chunk shape of the original element.
    scale_factors
        The multiscale pyramid of the written elements, as in :meth:`spatialdata.models.Image2DModel.parse`. By
        default each element gets the pyramid of the original.

    Returns
    -------
    The store holding the aligned elements (`sdata_path`, or `out_path`), read back.

    Examples
    --------
    >>> sdata = align_to_cs("sample.zarr", "he_image")
    >>> loader = Loader(images=["morphology_focus", "he_image_aligned"], tile_dim_in_units=224).add_sdata(sdata)
    """
    names = [elements] if isinstance(elements, str) else list(elements)
    if len(set(names)) != len(names):
        raise ValueError(f"`elements` has duplicates: {names}.")
    source_path = Path(sdata_path)
    source = read_zarr(source_path)
    new_store = out_path is not None
    if not new_store and not suffix:
        raise ValueError("`suffix` must not be empty when the aligned elements are added to the source store.")

    specs: list[_AlignmentSpec] = []
    for name in names:
        target = name if new_store else f"{name}{suffix}"
        spec = _alignment_spec(source, name, cs=cs, target=target, like=like, scale_factors=scale_factors)
        if spec is None:
            continue
        if not new_store and target in source and not overwrite:
            raise ValueError(f"`{target}` already exists in {source_path}; pass `overwrite=True` to replace it.")
        specs.append(spec)

    if new_store:
        destination_path = Path(cast("str | PathLike[str]", out_path))
        if destination_path.resolve() == source_path.resolve():
            raise ValueError(
                "`out_path` must differ from `sdata_path`; pass `out_path=None` to add to the source store."
            )
        if destination_path.exists():
            if not overwrite:
                raise ValueError(f"{destination_path} exists; pass `overwrite=True` to replace it.")
            shutil.rmtree(destination_path)
        shutil.copytree(source_path, destination_path)
        destination = read_zarr(destination_path)
    else:
        destination_path, destination = source_path, source

    for spec in specs:
        aligned = _aligned_element(source, spec, cs, chunk)
        if spec.target in destination:
            # remove the stale (copied or previous) element: first from memory, since spatialdata refuses to delete
            # files that back an element it holds, then from disk. `aligned` is never read from these files: it is
            # computed from `source`, which is another store (new-store mode) or another element (same-store mode).
            del destination[spec.target]
            destination.delete_element_from_disk(spec.target)
        destination[spec.target] = aligned
        destination.write_element(spec.target)

    written = read_zarr(destination_path)
    for spec in specs:
        _check_written(written, spec, cs)
    return written
