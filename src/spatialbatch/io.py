from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from annbatch.io import DATASET_PREFIX, V1_ENCODING, DatasetCollection, write_sharded
from annbatch.utils import load_all_aligned
from tqdm.auto import tqdm
from zarr.codecs import ZstdCodec

from .utils import (
    TileGeometry,
    aligned_tiles,
    element_to_cs_matrix,
    is_identity_xy,
    morton_order,
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
    - ``images/<name>`` and ``labels/<name>``: the full-resolution arrays as sharded zarr v3 arrays, chunked so that
      a tile touches few chunks and sharded so that a request is a few large reads.

    Only elements whose pixel grid is the coordinate system `cs` (no rotation or scaling) can be written, because a
    tile is then a plain slice. Elements such as a rotated H&E image are tiled in place instead, see
    :meth:`spatialbatch.Loader.add_sdata`.

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
    images, labels
        Names of the elements to write.
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
    validate_inputs(sdata, region, images, labels, cs, table_name)
    for name in (*images, *labels):
        if not is_identity_xy(element_to_cs_matrix(sdata[name], cs)):
            raise NotImplementedError(
                f"`{name}` is not in the pixel grid of `{cs}`, so its tiles are not slices and it cannot be written to a "
                "collection. Tile it in place with `Loader.add_sdata` instead."
            )
    table, tiles = aligned_tiles(
        sdata, region, table_name, cs, tile_scale=tile_scale, tile_dim_in_units=tile_dim_in_units
    )
    geometry = tile_starts(tiles)
    order = morton_order(tiles["x"].to_numpy(), tiles["y"].to_numpy(), block_px)

    adata = table[order].copy()
    adata.uns.pop("spatialdata_attrs", None)  # refers to the element names of the source, not to the collection
    adata.obsm["tile_origin"] = np.stack([geometry.x0, geometry.y0], axis=1)[order].astype(np.int32)
    write_sharded(group, adata, n_obs_per_chunk=min(n_obs_per_chunk, adata.shape[0]), shard_size=shard_size, key=key)

    out = group[key]
    for kind, names in (("images", images), ("labels", labels)):
        for name in names:
            _copy_sharded(
                scale0(str(sdata.path), kind, name),
                out.require_group(kind),
                name,
                chunk=image_chunk,
                shard=image_shard,
                compressors=compressors,
            )
    out.update_attributes(
        {
            ENCODING_KEY: {
                "encoding-version": "0.1.0",
                "tile_hw": [geometry.height, geometry.width],
                "cs": cs,
                "region": region,
                "images": list(images),
                "labels": list(labels),
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
    sources: dict[str, _Source] = {}
    for kind, names in (("images", images), ("labels", labels)):
        for name in names:
            array = group[kind][name]
            sources[name] = _Source(kind, array, array.shape[0] if kind == "images" else None, array.dtype)
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
