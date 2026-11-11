from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import zarr
from pandas import CategoricalDtype
from spatialdata import get_centroids, get_element_instances, to_circles
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    get_axes_names,
    get_model,
    get_table_keys,
)
from spatialdata.transformations import get_transformation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from geopandas import GeoDataFrame
    from spatialdata import SpatialData

    from spatialbatch.types import SpatialLoaderOutput

__all__ = [
    "TileGeometry",
    "aligned_tiles",
    "element_to_cs_matrix",
    "get_tile_coords",
    "integer_offset_xy",
    "is_identity_xy",
    "level_array",
    "level_to_cs_matrices",
    "morton_order",
    "rasterize_level",
    "round_half_up",
    "scale0",
    "tile_starts",
    "to_spatialstream_batch",
    "validate_inputs",
]


def validate_inputs(
    sdata: SpatialData,
    region: str,
    images: Sequence[str],
    labels: Sequence[str],
    cs: str,
    table_name: str,
) -> None:
    """Validate that a region, its pixel elements and its table can be loaded together.

    Ported from ``XeniumDataset._validate``: same checks, same messages.

    Raises
    ------
    ValueError
        If the region is a points element, a pixel element has the wrong model, or `cs` is not a coordinate system
        of the region and of every pixel element.
    TypeError
        If the region column of the table is not categorical.
    RuntimeError
        If instances of the region are not annotated by the table.
    """
    region_elem = sdata[region]
    if get_model(region_elem) == PointsModel:
        raise ValueError("`regions_element` must be a shapes or labels element, points are currently not supported.")
    region_trans = get_transformation(region_elem, get_all=True)
    for name in images:
        if get_model(sdata[name]) not in (Image2DModel, Image3DModel):
            raise ValueError("`images_element` must be an image element.")
    for name in labels:
        if get_model(sdata[name]) not in (Labels2DModel, Labels3DModel):
            raise ValueError("`labels_element` must be a labels element.")
    for name in (*images, *labels):
        elem_trans = get_transformation(sdata[name], get_all=True)
        if not (cs in region_trans and cs in elem_trans):
            raise ValueError(f"The coordinate system `{cs}` is not valid for the region `{region}` and image `{name}`.")

    table = sdata.tables[table_name]
    _, region_key, instance_key = get_table_keys(table)
    if not isinstance(table.obs[region_key].dtype, CategoricalDtype):
        raise TypeError(
            f"The `regions_element` column `{region_key}` in the table must be a categorical dtype. Please convert it."
        )
    if get_model(region_elem) in (Labels2DModel, Labels3DModel):
        indices = get_element_instances(region_elem).tolist()
    else:
        indices = region_elem.index.tolist()
    instance_column = table.obs.loc[table.obs[region_key] == region, instance_key]
    if not set(indices).issubset(instance_column):
        raise RuntimeError(
            "Some of the instances in the region element are not annotated by the table. You can remove some "
            "instances from the region element or add them to the table."
        )


def get_tile_coords(
    circles: GeoDataFrame,
    cs: str,
    *,
    tile_scale: float | None = 1.0,
    tile_dim_in_units: float | None = None,
) -> pd.DataFrame:
    """Get the centroid of each region in `cs` and the extent of its tile.

    Port of ``spatialstream.dataloaders.xenium._get_tile_coords`` with the same numbers, but without its side effects:
    the original assigns ``circles.radius`` in place (so building a second dataset from the same ``sdata`` compounds
    ``tile_scale``) and calls ``transform`` twice discarding the result. Neither changes the returned coordinates, so
    both are dropped. As in the original, the extent is read as a number of units of `cs`.

    Returns
    -------
    A dataframe indexed like `circles` with columns ``[x, y, extent, minx, miny, maxx, maxy]``.
    """
    if tile_dim_in_units is not None:
        radius = np.full(len(circles), tile_dim_in_units / 2, dtype=np.float64)
    else:
        radius = circles.radius.to_numpy(dtype=np.float64) * tile_scale
    extent = (radius * 2).reshape(-1, 1)
    centroids = get_centroids(circles, coordinate_system=cs)
    axes = get_axes_names(centroids)
    centroids_numpy = np.asarray((centroids.compute() if hasattr(centroids, "compute") else centroids).values)
    min_coordinates = centroids_numpy - extent / 2
    max_coordinates = centroids_numpy + extent / 2
    return pd.DataFrame(
        np.hstack([centroids_numpy, extent, min_coordinates, max_coordinates]),
        columns=[*axes, "extent", *("min" + ax for ax in axes), *("max" + ax for ax in axes)],
        index=circles.index,
    )


def aligned_tiles(
    sdata: SpatialData,
    region: str,
    table_name: str,
    cs: str,
    *,
    tile_scale: float | None = 1.0,
    tile_dim_in_units: float | None = None,
) -> tuple[Any, pd.DataFrame]:
    """The table of `sdata` and the tile of each of its rows (see :func:`get_tile_coords`).

    Rows are matched to regions by instance id, as ``XeniumDataset`` does with ``pd.Index.get_indexer``.

    Raises
    ------
    NotImplementedError
        If the table annotates more than `region`.
    RuntimeError
        If instances of `region` are not annotated by the table.
    """
    table = sdata.tables[table_name]
    _, region_key, instance_key = get_table_keys(table)
    if not (table.obs[region_key] == region).all():
        raise NotImplementedError(
            f"The table `{table_name}` annotates more than `{region}`; subset it to that region first."
        )
    tiles = get_tile_coords(to_circles(sdata[region]), cs, tile_scale=tile_scale, tile_dim_in_units=tile_dim_in_units)
    positions = pd.Index(tiles.index).get_indexer(table.obs[instance_key])
    if (positions < 0).any() or len(positions) != len(tiles):
        raise RuntimeError(f"Some instances of region `{region}` are not annotated by the table.")
    return table, tiles.iloc[positions]


def scale0(sdata_path: str, kind: Literal["images", "labels"], name: str) -> zarr.Array:
    """The full-resolution array of a multiscale image / labels element of a zarr-backed SpatialData."""
    return level_array(sdata_path, kind, name, 0)


def level_array(sdata_path: str, kind: Literal["images", "labels"], name: str, level: int) -> zarr.Array:
    """Pyramid level `level` (0 is the full resolution) of an image / labels element of a zarr-backed SpatialData."""
    group = zarr.open_group(sdata_path, mode="r")[f"{kind}/{name}"]
    attrs = dict(group.attrs)
    return group[attrs.get("ome", attrs)["multiscales"][0]["datasets"][level]["path"]]


def element_to_cs_matrix(element: Any, cs: str) -> np.ndarray:
    """The 3x3 affine matrix taking ``(x, y, 1)`` in the element's pixel grid to the coordinate system `cs`."""
    transformation = get_transformation(element, to_coordinate_system=cs)
    return np.asarray(transformation.to_affine_matrix(input_axes=("x", "y"), output_axes=("x", "y")), dtype=np.float64)


def level_to_cs_matrices(element: Any, cs: str) -> list[np.ndarray]:
    """:func:`element_to_cs_matrix` of every pyramid level of `element`, finest first (one entry if not multiscale)."""
    if not hasattr(element, "keys"):  # a single-scale DataArray
        return [element_to_cs_matrix(element, cs)]
    keys = sorted(element.keys(), key=lambda k: int(str(k).removeprefix("scale")))
    return [element_to_cs_matrix(next(iter(element[k].ds.values())), cs) for k in keys]


def rasterize_level(matrices: Sequence[np.ndarray], extent: float, unit_to_pixels: float) -> int:
    """The pyramid level :func:`spatialdata.rasterize` reads a tile of side `extent` (units of `cs`) from.

    That is the coarsest level that still has at least ``sqrt(2)`` level pixels per output pixel along both axes, or
    the full resolution if none has. The criterion is linear in `extent`, so the level is the same for every tile.
    """
    target = extent * unit_to_pixels * np.sqrt(2)
    corners = np.array([[0.0, extent, 0.0, extent], [0.0, 0.0, extent, extent], [1.0, 1.0, 1.0, 1.0]])
    for level in reversed(range(len(matrices))):
        in_level = np.linalg.inv(matrices[level]) @ corners
        if (np.ptp(in_level[:2], axis=1) >= target).all():
            return level
    return 0


def is_identity_xy(matrix: np.ndarray, atol: float = 1e-9) -> bool:
    """Whether a pixel grid coincides with `cs`, i.e. a tile is a plain slice of the stored array."""
    return bool(np.allclose(matrix, np.eye(3), atol=atol))


def integer_offset_xy(matrix: np.ndarray, atol: float = 1e-9) -> tuple[int, int] | None:
    """The ``(x, y)`` offset if a pixel grid is `cs` shifted by whole pixels (tiles are slices of it), else ``None``."""
    offset = matrix[:2, 2]
    shifted = np.eye(3)
    shifted[:2, 2] = np.round(offset)
    if not np.allclose(matrix, shifted, atol=atol):
        return None
    return int(shifted[0, 2]), int(shifted[1, 2])


def round_half_up(a: np.ndarray) -> np.ndarray:
    """``floor(a + 0.5)`` as integers. This is the pixel `spatialdata` starts a tile at (verified against it)."""
    return np.floor(np.asarray(a, dtype=np.float64) + 0.5).astype(np.int64)


@dataclass(frozen=True)
class TileGeometry:
    """Where each tile of one SpatialData object lies, in the pixel grid of an array whose grid is `cs`.

    Attributes
    ----------
    x0, y0
        First pixel of each tile (may be negative, or run past the array, for cells near the border).
    height, width
        Size of every tile in pixels.
    """

    x0: np.ndarray
    y0: np.ndarray
    height: int
    width: int

    def __len__(self) -> int:
        return len(self.x0)


def tile_starts(tiles: pd.DataFrame, unit_to_pixels: float = 1.0) -> TileGeometry:
    """Tile geometry from :func:`get_tile_coords` in the pixel grid of `cs` with `unit_to_pixels` pixels per unit.

    With the default of 1, pixel ``(x, y)`` of the grid is the unit square at ``(x, y)`` of `cs`.

    Raises
    ------
    ValueError
        If the extent is not one integer number of pixels for all tiles, so tiles could not be batched.
    """
    extent = tiles["extent"].to_numpy() * unit_to_pixels
    size = int(round(float(extent[0])))
    if not np.allclose(extent, size) or size < 1:
        raise ValueError(
            "Tiles must all have the same integer size in pixels to be batched, but the extents range over "
            f"[{extent.min():.3f}, {extent.max():.3f}] pixels. Pass `tile_dim_in_units` such that it is an integer "
            "number of pixels."
        )
    return TileGeometry(
        x0=round_half_up(tiles["minx"].to_numpy() * unit_to_pixels),
        y0=round_half_up(tiles["miny"].to_numpy() * unit_to_pixels),
        height=size,
        width=size,
    )


def morton_order(x: np.ndarray, y: np.ndarray, block: int = 1024) -> np.ndarray:
    """Permutation sorting points along a Z-order curve over a grid of ``block``-sized cells.

    Cells that are close in the plane end up close in the order, so a run of consecutive cells reads a compact region
    of the image. Ties inside one grid cell keep their original order.
    """
    gx = np.maximum(np.floor_divide(np.asarray(x, dtype=np.float64), block), 0).astype(np.uint64)
    gy = np.maximum(np.floor_divide(np.asarray(y, dtype=np.float64), block), 0).astype(np.uint64)

    def spread(v: np.ndarray) -> np.ndarray:
        v = v & np.uint64(0xFFFFFFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x0000FFFF0000FFFF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x00FF00FF00FF00FF)
        v = (v | (v << np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
        v = (v | (v << np.uint64(2))) & np.uint64(0x3333333333333333)
        v = (v | (v << np.uint64(1))) & np.uint64(0x5555555555555555)
        return v

    return np.argsort(spread(gx) | (spread(gy) << np.uint64(1)), kind="stable")


def to_spatialstream_batch(
    batch: SpatialLoaderOutput,
    *,
    image: str,
    annotations: str | list[str] | None = None,
) -> dict[str, Any]:
    """Reshape a batch to what ``XeniumDataset`` + default collation produce.

    The result has the keys ``image``, ``gexp``, ``coord`` and (if `annotations` is given) ``annotations``, with the
    shapes ``(B, C, H, W)``, ``(B, 1, n_var)``, ``(B, 1, 2)`` and ``(B, 1, n_annotations)``.
    """
    X = batch["X"]
    gexp = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    out: dict[str, Any] = {"image": batch["images"][image], "gexp": gexp[:, None, :]}
    if batch["coords"] is not None:
        out["coord"] = batch["coords"][:, None, :]
    if annotations is not None:
        cols = [annotations] if isinstance(annotations, str) else list(annotations)
        if batch["obs"] is None:
            raise ValueError("The batch has no obs, so there are no annotations to return.")
        out["annotations"] = batch["obs"][cols].to_numpy()[:, None, :]
    return out
