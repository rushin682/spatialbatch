from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from annbatch.io import DATASET_PREFIX, V1_ENCODING, DatasetCollection, write_sharded
from annbatch.utils import load_all_aligned
from spatialdata.models import get_axes_names
from tqdm.auto import tqdm
from zarr.codecs import ZstdCodec

from .utils import (
    TileGeometry,
    aligned_tiles,
    integer_offset_xy,
    level_array,
    level_to_cs_matrices,
    morton_order,
    rasterize_level,
    scale0,
    tile_starts,
    validate_inputs,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from os import PathLike

    import pandas as pd
    from spatialdata import SpatialData
    from zarr.abc.codec import BytesBytesCodec

    from .loader import _Item

__all__ = ["SpatialDatasetCollection", "write_sharded_sdata"]

#: Key of the group attribute that describes how a dataset of a collection was written.
ENCODING_KEY = "spatialbatch"


def _copy_sharded(
    src: zarr.Array,
    group: zarr.Group,
    name: str,
    *,
    chunk: int,
    shard: int,
    compressors: Iterable[BytesBytesCodec],
) -> None:
    """Copy `src` to a sharded array, one shard-sized block at a time so that memory use stays at ~one shard."""
    lead = src.shape[:-2]
    height, width = src.shape[-2:]
    dst = group.create_array(
        name,
        shape=src.shape,
        dtype=src.dtype,
        chunks=(*lead, min(chunk, height), min(chunk, width)),
        shards=(*lead, min(shard, height), min(shard, width)),
        compressors=tuple(compressors),
        fill_value=0,
        overwrite=True,
    )
    blocks = [(y, x) for y in range(0, height, shard) for x in range(0, width, shard)]
    for y, x in tqdm(blocks, desc=f"Writing {name}", leave=False):
        dst[..., y : y + shard, x : x + shard] = src[..., y : y + shard, x : x + shard]


# Side in pixels of the blocks a shard is resampled in, which bounds the source region read at once.
_RESAMPLE_BLOCK = 1024


def _resample_sharded(
    src: zarr.Array,
    src_to_cs: np.ndarray,
    unit_to_pixels: float,
    box: tuple[int, int, int, int],
    group: zarr.Group,
    name: str,
    *,
    chunk: int,
    shard: int,
    compressors: Iterable[BytesBytesCodec],
) -> None:
    """Resample `src` into the pixel grid of `cs` over `box` and write it as a sharded array, one shard per thread.

    Pixel ``(x, y)`` of the grid takes the value of `src` at the point ``(x, y) / unit_to_pixels`` of `cs`, by nearest
    neighbour, which is how :func:`spatialdata.rasterize` samples a tile with ``target_unit_to_pixels=unit_to_pixels``.
    Pixels whose point is outside `src` are 0, as in ``rasterize``.

    Parameters
    ----------
    src
        A 2D labels array ``(y, x)`` or image array ``(c, y, x)``.
    src_to_cs
        3x3 affine matrix taking ``(x, y, 1)`` in the pixel grid of `src` to `cs`.
    box
        ``(x0, y0, x1, y1)``, the region of the grid to write; ``(x0, y0)`` becomes pixel ``(0, 0)`` of the array.
    """
    from scipy.ndimage import affine_transform

    gx0, gy0, gx1, gy1 = box
    lead = src.shape[:-2]
    src_h, src_w = src.shape[-2:]
    height, width = gy1 - gy0, gx1 - gx0
    dst = group.create_array(
        name,
        shape=(*lead, height, width),
        dtype=src.dtype,
        chunks=(*lead, min(chunk, height), min(chunk, width)),
        shards=(*lead, min(shard, height), min(shard, width)),
        compressors=tuple(compressors),
        fill_value=0,
        overwrite=True,
    )
    cs_to_src = np.linalg.inv(src_to_cs)
    swap = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # (x, y, 1) <-> (row, col, 1)

    def resample_block(out: np.ndarray, x: int, y: int) -> None:
        """Fill `out`, the pixels from ``(x, y)`` (relative to `box`) of the grid."""
        bh, bw = out.shape[-2:]
        grid_to_cs = np.array([[1, 0, gx0 + x], [0, 1, gy0 + y], [0, 0, 1]], dtype=np.float64) / np.array(
            [[unit_to_pixels], [unit_to_pixels], [1.0]]
        )
        to_src = cs_to_src @ grid_to_cs  # (x, y, 1) of `out` -> (x, y, 1) of `src`
        corners = to_src @ np.array([[0, bw - 1, 0, bw - 1], [0, 0, bh - 1, bh - 1], [1, 1, 1, 1]], dtype=np.float64)
        # Read the part of `src` the block samples from, with a margin so that rounding to the nearest pixel and
        # the bounds check of `affine_transform` see the same neighbourhood as on the whole array.
        sx0, sy0 = (max(int(np.floor(v)) - 2, 0) for v in corners[:2].min(axis=1))
        sx1, sy1 = (min(int(np.ceil(v)) + 3, n) for v, n in zip(corners[:2].max(axis=1), (src_w, src_h), strict=True))
        if sx0 >= sx1 or sy0 >= sy1:
            return  # the block lies outside `src`: it stays 0
        region = np.asarray(src[..., sy0:sy1, sx0:sx1])
        rc = swap @ to_src @ swap  # (row, col, 1) of `out` -> (row, col, 1) of `src`
        offset = rc[:2, 2] - np.array([sy0, sx0])
        planes = region.reshape(-1, *region.shape[-2:])
        for plane, target in zip(planes, out.reshape(-1, bh, bw), strict=True):
            affine_transform(
                plane, rc[:2, :2], offset=offset, output=target, order=0, mode="constant", cval=0, prefilter=False
            )

    def write_shard(origin: tuple[int, int]) -> None:
        y, x = origin
        out = np.zeros((*lead, min(shard, height - y), min(shard, width - x)), dtype=src.dtype)
        for by in range(0, out.shape[-2], _RESAMPLE_BLOCK):
            for bx in range(0, out.shape[-1], _RESAMPLE_BLOCK):
                resample_block(out[..., by : by + _RESAMPLE_BLOCK, bx : bx + _RESAMPLE_BLOCK], x + bx, y + by)
        dst[..., y : y + out.shape[-2], x : x + out.shape[-1]] = out

    shards = [(y, x) for y in range(0, height, shard) for x in range(0, width, shard)]
    n_threads = zarr.config.get("threading.max_workers") or os.cpu_count() or 1
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        for _ in tqdm(pool.map(write_shard, shards), total=len(shards), desc=f"Resampling {name}", leave=False):
            pass


def _resampled_box(
    src_to_cs: np.ndarray, src_hw: tuple[int, int], unit_to_pixels: float, geometry: TileGeometry
) -> tuple[int, int, int, int]:
    """The part of the grid that both `src` and some tile cover, as ``(x0, y0, x1, y1)`` (never empty)."""
    src_h, src_w = src_hw
    edges = np.array([[-0.5, src_w - 0.5, -0.5, src_w - 0.5], [-0.5, -0.5, src_h - 0.5, src_h - 0.5], [1, 1, 1, 1]])
    in_grid = (src_to_cs @ edges)[:2] * unit_to_pixels
    x0 = max(int(np.floor(in_grid[0].min())) - 1, int(geometry.x0.min()))
    y0 = max(int(np.floor(in_grid[1].min())) - 1, int(geometry.y0.min()))
    x1 = min(int(np.ceil(in_grid[0].max())) + 2, int(geometry.x0.max()) + geometry.width)
    y1 = min(int(np.ceil(in_grid[1].max())) + 2, int(geometry.y0.max()) + geometry.height)
    if x0 >= x1 or y0 >= y1:  # no tile overlaps the element: keep one pixel, which tiles never reach
        return x0, y0, x0 + 1, y0 + 1
    return x0, y0, x1, y1


def write_sharded_sdata(
    group: zarr.Group,
    sdata: SpatialData,
    *,
    key: str,
    region: str = "cell_circles",
    images: Sequence[str] = ("morphology_focus",),
    labels: Sequence[str] = (),
    table_name: str = "table",
    cs: str = "global",
    tile_dim_in_units: float | None = None,
    tile_scale: float = 1.0,
    target_unit_to_pixels: float = 1.0,
    block_px: int = 1024,
    n_obs_per_chunk: int = 64,
    shard_size: int | str = "1GB",
    image_chunk: int = 256,
    image_shard: int = 4096,
    compressors: Iterable[BytesBytesCodec] = (ZstdCodec(level=3),),
) -> None:
    """Write one backed :class:`spatialdata.SpatialData` as a dataset of a spatialbatch collection.

    The counterpart of :func:`annbatch.write_sharded`. What it writes at ``group[key]``:

    - the table, with its rows sorted along a Morton (Z-order) curve of the tile centers so that consecutive rows
      are spatially close, written with :func:`annbatch.write_sharded`, plus ``obsm["tile_origin"]``, the first
      pixel ``(x, y)`` of each tile;
    - ``images/<name>`` and ``labels/<name>``: every element in one common pixel grid, the *target grid*, as sharded
      zarr v3 arrays, chunked so that a tile touches few chunks and sharded so that a request is a few large reads.

    The target grid is `cs` with `target_unit_to_pixels` pixels per unit, so that every tile is a plain slice of
    every array. Each element gets there in the cheapest exact way:

    - an element whose pixel grid already is the target grid is copied unchanged, at full resolution;
    - one that is shifted from it by whole pixels is copied unchanged too, and its offset is stored;
    - any other one (rotated, scaled, sheared, or shifted by a fraction of a pixel, such as an H&E image registered
      to the morphology image) is resampled once into the target grid, over the region that tiles can reach.

    Resampling takes, for each target pixel, the nearest pixel of the pyramid level that
    :func:`spatialdata.rasterize` reads with ``target_unit_to_pixels``, which is what ``XeniumDataset`` and
    :meth:`spatialbatch.Loader.add_sdata` return with ``rasterize=True``. The one difference: tiles start on the target
    grid, so a tile may be shifted by up to half a target pixel from the one ``rasterize`` cuts at the cell's exact
    position. Labels stay labels, as no values are interpolated.

    Parameters
    ----------
    group
        The destination group, must be zarr v3.
    sdata
        A SpatialData read from zarr, whose table annotates only `region`.
    key
        The name of the dataset in `group`, ``dataset_<i>`` for it to be found by :class:`SpatialDatasetCollection`.
    region, table_name, cs, tile_dim_in_units, tile_scale
        See :meth:`spatialbatch.Loader.add_sdata` and :class:`spatialbatch.Loader`.
    target_unit_to_pixels
        Pixels per unit of `cs` of the target grid. A tile is ``tile_dim_in_units * target_unit_to_pixels`` pixels
        wide, which must be an integer. With the default of 1, an element whose pixel grid is `cs` (Xenium's
        ``morphology_focus`` in ``global``) is copied as it is.
    images, labels
        Names of the elements to write, 2D images ``(c, y, x)`` and labels ``(y, x)``.
    block_px
        Side in pixels of the grid used to order cells. Match it to the window the loader reads (1024 by default).
    n_obs_per_chunk, shard_size
        Chunking of the table, see :func:`annbatch.write_sharded`.
    image_chunk, image_shard
        Side in pixels of the chunks and of the shards of the images and labels. `image_shard` must be divisible by
        `image_chunk`.
    compressors
        Compressors for the images and labels. The table uses the :func:`annbatch.write_sharded` default.
    """
    if not sdata.is_backed():
        raise ValueError("`sdata` must be backed by a zarr store (use `spatialdata.read_zarr`).")
    if image_shard % image_chunk != 0:
        raise ValueError(f"image_shard ({image_shard}) must be divisible by image_chunk ({image_chunk}).")
    if not target_unit_to_pixels > 0:
        raise ValueError(f"target_unit_to_pixels must be positive, got {target_unit_to_pixels}.")
    validate_inputs(sdata, region, images, labels, cs, table_name)
    for kind, names in (("images", images), ("labels", labels)):
        for name in names:
            if get_axes_names(sdata[name]) != (("c", "y", "x") if kind == "images" else ("y", "x")):
                raise NotImplementedError(f"Only 2D {kind} are supported, but `{name}` has axes {sdata[name].dims}.")
    table, tiles = aligned_tiles(
        sdata, region, table_name, cs, tile_scale=tile_scale, tile_dim_in_units=tile_dim_in_units
    )
    geometry = tile_starts(tiles, target_unit_to_pixels)
    order = morton_order(
        tiles["x"].to_numpy() * target_unit_to_pixels, tiles["y"].to_numpy() * target_unit_to_pixels, block_px
    )

    adata = table[order].copy()
    adata.uns.pop("spatialdata_attrs", None)  # refers to the element names of the source, not to the collection
    adata.obsm["tile_origin"] = np.stack([geometry.x0, geometry.y0], axis=1)[order].astype(np.int32)
    write_sharded(group, adata, n_obs_per_chunk=min(n_obs_per_chunk, adata.shape[0]), shard_size=shard_size, key=key)

    out = group[key]
    origins: dict[str, list[int]] = {}
    resampled: dict[str, int] = {}
    extent = float(tiles["extent"].iloc[0])
    to_grid = np.diag([target_unit_to_pixels, target_unit_to_pixels, 1.0])  # cs -> target grid
    for kind, names in (("images", images), ("labels", labels)):
        for name in names:
            matrices = level_to_cs_matrices(sdata[name], cs)
            offset = integer_offset_xy(to_grid @ matrices[0])
            if offset is not None:  # tiles are slices of the stored array: copy it
                src = scale0(str(sdata.path), kind, name)
                _copy_sharded(
                    src, out.require_group(kind), name, chunk=image_chunk, shard=image_shard, compressors=compressors
                )
            else:
                level = rasterize_level(matrices, extent, target_unit_to_pixels)
                src = level_array(str(sdata.path), kind, name, level)
                box = _resampled_box(matrices[level], src.shape[-2:], target_unit_to_pixels, geometry)
                _resample_sharded(
                    src,
                    matrices[level],
                    target_unit_to_pixels,
                    box,
                    out.require_group(kind),
                    name,
                    chunk=image_chunk,
                    shard=image_shard,
                    compressors=compressors,
                )
                offset, resampled[name] = (box[0], box[1]), level
            origins[name] = list(offset)
    out.update_attributes(
        {
            ENCODING_KEY: {
                "encoding-version": "0.2.0",
                "tile_hw": [geometry.height, geometry.width],
                "cs": cs,
                "target_unit_to_pixels": target_unit_to_pixels,
                "region": region,
                "images": list(images),
                "labels": list(labels),
                "origins": origins,
                "resampled_from_level": resampled,
                "block_px": block_px,
                "source": str(sdata.path),
            }
        }
    )


def open_dataset_group(
    group: zarr.Group,
    images: Sequence[str],
    labels: Sequence[str],
    annotations: Sequence[str] | None,
) -> tuple[Any, pd.DataFrame | None, pd.DataFrame, _Item]:
    """Open a dataset written by :func:`write_sharded_sdata` for :class:`spatialbatch.Loader`."""
    from .loader import _Item, _Source

    meta = group.attrs.get(ENCODING_KEY)
    if meta is None:
        raise ValueError(f"{group.name} was not written by `write_sharded_sdata`.")
    missing = [n for n in images if n not in meta["images"]] + [n for n in labels if n not in meta["labels"]]
    if missing:
        raise ValueError(
            f"{missing} are not in the collection, which has images {meta['images']} and labels {meta['labels']}."
        )
    adata = load_all_aligned(group)
    origin = np.asarray(adata.obsm["tile_origin"][...])
    height, width = meta["tile_hw"]
    origins = meta.get("origins", {})  # absent before encoding 0.2.0, when every array started at pixel (0, 0)
    sources: dict[str, _Source] = {}
    for kind, names in (("images", images), ("labels", labels)):
        for name in names:
            array = group[kind][name]
            channels = array.shape[0] if kind == "images" else None
            x0, y0 = origins.get(name, (0, 0))
            sources[name] = _Source(kind, array, channels, array.dtype, origin=(int(x0), int(y0)))
    obs = adata.obs if annotations is None else adata.obs[list(annotations)]
    coords = np.asarray(adata.obsm["spatial"][...]) if "spatial" in adata.obsm else None
    item = _Item(
        TileGeometry(origin[:, 0].astype(np.int64), origin[:, 1].astype(np.int64), height, width), sources, coords
    )
    return adata.X, None if len(obs.columns) == 0 else obs, adata.var, item


class SpatialDatasetCollection(DatasetCollection):
    """A collection of spatially sorted, sharded SpatialData datasets, for :meth:`spatialbatch.Loader.use_collection`.

    This is :class:`annbatch.DatasetCollection` with a different writer. Iteration over the datasets,
    :attr:`is_empty` and :meth:`obs` are inherited. A dataset is one SpatialData, written by
    :func:`write_sharded_sdata`: unlike annbatch there is no mixing of rows across files on disk, since pixels are not
    rows that can be moved around. Slides are mixed when sampling, as the chunks drawn by one request come from any
    dataset.

    Parameters
    ----------
    group
        The base location of the collection, a :class:`zarr.Group` or a path ending in ``.zarr``.
    mode
        The mode the store is opened with if `group` is a path.

    Examples
    --------
    >>> collection = SpatialDatasetCollection("collection.zarr")
    >>> collection.add_sdatas(["a.zarr", "b.zarr"], images=["morphology_focus"], tile_dim_in_units=224)
    >>> loader = Loader(images=["morphology_focus"], batch_size=32, shuffle=True).use_collection(collection)
    """

    def __init__(self, group: zarr.Group | str | PathLike[str], *, mode: str = "a"):
        super().__init__(group, mode=mode)  # type: ignore[arg-type]
        if not isinstance(self._group, zarr.Group):
            raise TypeError("A SpatialDatasetCollection is always zarr.")

    def add_adatas(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        """Not supported: a dataset of this collection holds pixels, written by :meth:`add_sdatas`."""
        raise TypeError("A SpatialDatasetCollection holds pixels too: use `add_sdatas`.")

    def add_sdatas(self, sdata_paths: Iterable[str | PathLike[str]], **kwargs: Any) -> None:
        """Write SpatialData stores to the collection, each as a new dataset.

        Parameters
        ----------
        sdata_paths
            Paths to SpatialData zarr stores.
        **kwargs
            Passed to :func:`write_sharded_sdata`.
        """
        import spatialdata as sd

        paths = list(sdata_paths)
        first = len(self._dataset_keys)
        for i, path in enumerate(tqdm(paths, desc="Adding to collection")):
            write_sharded_sdata(self._group, sd.read_zarr(path), key=f"{DATASET_PREFIX}_{first + i}", **kwargs)
            self._group.update_attributes(V1_ENCODING)
