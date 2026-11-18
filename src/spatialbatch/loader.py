from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import zarr
import zarr.core.sync as zsync
from annbatch import Loader as AnnbatchLoader
from annbatch.utils import check_lt_1, convert, load_all_aligned
from spatialdata import bounding_box_query
from spatialdata import rasterize as spatialdata_rasterize
from xarray import DataArray, DataTree

from spatialbatch.samplers import RandomSampler, SequentialSampler

from .utils import (
    TileGeometry,
    aligned_tiles,
    element_to_cs_matrix,
    is_identity_xy,
    scale0,
    tile_starts,
    validate_inputs,
)

if TYPE_CHECKING:
    from collections import OrderedDict
    from collections.abc import Callable, Iterator, Mapping, Sequence

    import pandas as pd
    from spatialdata import SpatialData

    from spatialbatch.io import SpatialDatasetCollection
    from spatialbatch.types import SpatialLoaderOutput

__all__ = ["Loader"]

# Side of the grid used to group the tiles of one request into windows that are each read from disk once.
_WINDOW_PX = 1024


@dataclass
class _Source:
    """One pixel modality (an image or a labels element) of one SpatialData object."""

    kind: Literal["images", "labels"]
    array: zarr.Array | None  # scale 0; ``None`` when tiles must be produced by `fallback`
    channels: int | None  # ``None`` for labels
    dtype: np.dtype
    fallback: Callable[[int], np.ndarray] | None = None
    origin: tuple[int, int] = (0, 0)  # pixel (x, y) of the tile grid that is pixel (0, 0) of `array`

    @property
    def signature(self) -> tuple[str, int | None, np.dtype]:
        return self.kind, self.channels, self.dtype


@dataclass
class _Item:
    """Everything the loader keeps about one SpatialData object besides its table ``X`` / ``obs`` / ``var``."""

    geometry: TileGeometry
    sources: dict[str, _Source]
    coords: np.ndarray | None


class _SpatialdataTile:
    """Produce one tile with :func:`spatialdata.rasterize` / :func:`spatialdata.bounding_box_query`, as ``XeniumDataset`` does.

    This is the path for elements whose pixel grid is not the coordinate system (a rotated H&E image, say) or when
    resampling is requested: correct for every transformation, but one spatialdata call per tile. A
    :class:`~spatialbatch.SpatialDatasetCollection` resamples such elements once when it is written instead.
    A class rather than a closure so that the loader can be pickled into :class:`torch.utils.data.DataLoader` workers.
    """

    def __init__(
        self,
        element: DataArray | DataTree,
        bounds: np.ndarray,
        cs: str,
        rasterize: bool,
        rasterize_kwargs: Mapping[str, Any],
    ):
        self.element, self.bounds, self.cs = element, bounds, cs
        self.rasterize, self.rasterize_kwargs = rasterize, dict(rasterize_kwargs)

    def __call__(self, row: int) -> np.ndarray:
        minx, miny, maxx, maxy = self.bounds[row]
        kwargs = {
            "axes": ("x", "y"),
            "min_coordinate": np.array([minx, miny]),
            "max_coordinate": np.array([maxx, maxy]),
            "target_coordinate_system": self.cs,
        }
        tile = (
            spatialdata_rasterize(self.element, **kwargs, **self.rasterize_kwargs)
            if self.rasterize
            else bounding_box_query(self.element, **kwargs)
        )
        if isinstance(tile, DataTree):
            tile = next(iter(tile["scale0"].ds.values()))
        return np.asarray(tile.values)


class Loader(AnnbatchLoader):
    """A loader that streams multi-modal cell tiles from on-disk :class:`spatialdata.SpatialData` stores.

    This is :class:`annbatch.Loader` with pixels added. A *row* is a cell. For every row the loader yields the same
    things annbatch does (its ``X`` expression, ``obs`` and ``index``) plus a tile of every requested image and labels
    element, centered on the cell, so that all modalities of a batch are aligned along axis 0.

    It follows annbatch's recipe. A :class:`~annbatch.abc.Sampler` emits :class:`~annbatch.types.LoadRequest` s, each a
    set of chunks of consecutive rows; one request is read from disk **at once**, concurrently for ``X`` and for every
    modality, through async :mod:`zarr` (and `zarrs` if configured, see :doc:`zarr-configuration`); the in-memory
    result is shuffled and split into batches. No torch :class:`~torch.utils.data.DataLoader` is required, and when one
    is used it must be given ``batch_size=None``.

    Tiles that are a plain slice of the stored array (images in the pixel grid of the coordinate system, which is the
    case for Xenium ``morphology_focus``, ``morphology_mip`` and the labels) are read as a few large windows and
    cropped in memory. This yields exactly the pixels :func:`spatialdata.rasterize` returns (verified against
    ``XeniumDataset``), without resampling. Other elements, such as a rotated H&E image, are tiled with spatialdata
    when loading in place, one call per tile; a :class:`~spatialbatch.SpatialDatasetCollection` resamples them once
    when written, after which they are read like any other array.
    Where a tile crosses the border of the array the missing pixels are `pad_value`.

    Parameters
    ----------
    images
        Names of the image elements to tile.
    labels
        Names of the labels elements to tile.
    tile_scale
        Tile size as a multiple of the region's diameter, ignored if `tile_dim_in_units` is set.
        See :class:`spatialstream.dataloaders.XeniumDataset`.
    tile_dim_in_units
        Side of the tile in units of the coordinate system. All tiles of a loader must have the same integer size.
    rasterize
        Mirror of ``XeniumDataset``: whether tiles are produced with :func:`spatialdata.rasterize` (``True``) or
        :func:`spatialdata.bounding_box_query` (``False``). It only matters on the spatialdata path.
    rasterize_kwargs
        Keyword arguments for :func:`spatialdata.rasterize`. The fast path requires ``{"target_unit_to_pixels": 1.0}``
        if `rasterize` is ``True``.
    pad_value
        Value of pixels outside the array.
    return_annotations
        `obs` columns to keep and yield. By default all are.
    transform
        Called on every :class:`~spatialbatch.types.SpatialLoaderOutput` before it is yielded.
    batch_sampler
        If not provided, a :class:`~spatialbatch.samplers.RandomSampler` (if `shuffle`) or
        :class:`~annbatch.samplers.SequentialSampler` is built from the arguments below.
    chunk_size
        Number of consecutive cells per chunk. Mutually exclusive with `batch_sampler`. Defaults to 64.
    preload_nchunks
        Number of chunks per request. Mutually exclusive with `batch_sampler`. Defaults to 8.
    shuffle
        Whether to shuffle. Mutually exclusive with `batch_sampler`. Defaults to ``False``.
    batch_size
        Batch size to yield. Mutually exclusive with `batch_sampler`. Defaults to 32.
    drop_last
        Whether to drop the last incomplete batch. Mutually exclusive with `batch_sampler`. Defaults to ``False``.
    rng
        Random number generator for shuffling. Mutually exclusive with `batch_sampler`.
    return_index
        Whether to yield the position of each row in the loader.
    to
        Output library for `X` and for the tiles.

    Examples
    --------
    >>> from spatialbatch import Loader
    >>> loader = Loader(
    ...     images=["morphology_focus"],
    ...     labels=["cell_labels"],
    ...     tile_dim_in_units=224,
    ...     rasterize=True,
    ...     rasterize_kwargs={"target_unit_to_pixels": 1.0},
    ...     batch_size=32,
    ...     shuffle=True,
    ... ).add_sdata(sdata)
    >>> for batch in loader:
    ...     batch["images"]["morphology_focus"]  # (32, 1, 224, 224)
    """

    _COMMON_SAMPLER_ARGS = {
        "chunk_size": 64,
        "preload_nchunks": 8,
        "batch_size": 32,
        "drop_last": False,
    }

    _items: list[_Item]
    _signature: dict[str, tuple[str, int | None, np.dtype]] | None = None
    _tile_hw: tuple[int, int] | None = None

    def __init__(
        self,
        *,
        images: Sequence[str] = ("morphology_focus",),
        labels: Sequence[str] = (),
        tile_scale: float = 1.0,
        tile_dim_in_units: float | None = None,
        rasterize: bool = False,
        rasterize_kwargs: Mapping[str, Any] = MappingProxyType({}),
        pad_value: float = 0,
        return_annotations: str | Sequence[str] | None = None,
        transform: Callable[[SpatialLoaderOutput], Any] | None = None,
        batch_sampler: Any = None,
        chunk_size: int | None = None,
        preload_nchunks: int | None = None,
        shuffle: bool | None = None,
        batch_size: int | None = None,
        drop_last: bool | None = None,
        rng: np.random.Generator | None = None,
        return_index: bool = False,
        to: Literal["torch", "jax"] | None = None,
    ):
        if batch_sampler is None:
            core = {
                "chunk_size": chunk_size,
                "preload_nchunks": preload_nchunks,
                "batch_size": batch_size,
                "drop_last": drop_last,
            }
            resolved = {k: Loader._COMMON_SAMPLER_ARGS[k] if v is None else v for k, v in core.items()}
            batch_sampler = RandomSampler(**resolved, rng=rng) if shuffle else SequentialSampler(**resolved)
            super().__init__(batch_sampler=batch_sampler, return_index=return_index, preload_to_gpu=False, to=to)
        else:  # let annbatch reject sampler arguments that conflict with a custom sampler
            super().__init__(
                batch_sampler=batch_sampler,
                chunk_size=chunk_size,
                preload_nchunks=preload_nchunks,
                shuffle=shuffle,
                batch_size=batch_size,
                drop_last=drop_last,
                rng=rng,
                return_index=return_index,
                preload_to_gpu=False,
                to=to,
            )
        if not images and not labels:
            raise ValueError("At least one of `images` and `labels` must be given.")
        self._image_names, self._label_names = tuple(images), tuple(labels)
        self._tile_scale, self._tile_dim_in_units = tile_scale, tile_dim_in_units
        self._rasterize, self._rasterize_kwargs = rasterize, dict(rasterize_kwargs)
        self._pad_value = pad_value
        self._annotations = [return_annotations] if isinstance(return_annotations, str) else return_annotations
        self.transform = transform
        self._items = []

    # -- adding data ---------------------------------------------------------------------------------------------

    def add_sdata(
        self,
        sdata: SpatialData,
        *,
        region: str = "cell_circles",
        table_name: str = "table",
        cs: str = "global",
    ) -> Self:
        """Append the cells of a backed :class:`spatialdata.SpatialData` to this loader.

        Parameters
        ----------
        sdata
            A SpatialData read from zarr (``sdata.is_backed()``), whose table annotates only `region`.
            The table's rows are the loader's rows, in table order.
        region
            The shapes (or labels) element whose centroids are the tile centers.
        table_name
            The table with the expression (``X``), ``obs`` and ``obsm["spatial"]``.
        cs
            The coordinate system the tiles are cut in.
        """
        if self._num_open_iters > 0:
            raise RuntimeError(
                "Cannot add datasets while the loader is being iterated. Finish or close the iterator first."
            )
        if not sdata.is_backed():
            raise ValueError("`sdata` must be backed by a zarr store (use `spatialdata.read_zarr`) to be streamed.")
        validate_inputs(sdata, region, self._image_names, self._label_names, cs, table_name)

        _, tiles = aligned_tiles(
            sdata, region, table_name, cs, tile_scale=self._tile_scale, tile_dim_in_units=self._tile_dim_in_units
        )
        geometry = tile_starts(tiles)
        bounds = tiles[["minx", "miny", "maxx", "maxy"]].to_numpy()

        sources = self._open_sources(sdata, bounds, geometry, cs)
        group = zarr.open_group(sdata.path, mode="r")["tables"][table_name]
        adata = load_all_aligned(group)
        obs = adata.obs if self._annotations is None else adata.obs[self._annotations]
        coords = np.asarray(adata.obsm["spatial"][...]) if "spatial" in adata.obsm else None
        self._add_item(adata.X, None if len(obs.columns) == 0 else obs, adata.var, _Item(geometry, sources, coords))
        return self

    def _open_sources(
        self, sdata: SpatialData, bounds: np.ndarray, geometry: TileGeometry, cs: str
    ) -> dict[str, _Source]:
        sources: dict[str, _Source] = {}
        native = (not self._rasterize) or self._rasterize_kwargs == {"target_unit_to_pixels": 1.0}
        for kind, names in (("images", self._image_names), ("labels", self._label_names)):
            for name in names:
                element = sdata[name]
                array = scale0(str(sdata.path), kind, name)
                if array.ndim != (3 if kind == "images" else 2):
                    raise NotImplementedError(f"Only 2D {kind} are supported, but `{name}` has shape {array.shape}.")
                channels = array.shape[0] if kind == "images" else None
                if native and is_identity_xy(element_to_cs_matrix(element, cs)):
                    sources[name] = _Source(kind, array, channels, array.dtype)
                else:
                    fallback = _SpatialdataTile(element, bounds, cs, self._rasterize, self._rasterize_kwargs)
                    probe = fallback(0)
                    expected = (geometry.height, geometry.width)
                    if probe.shape[-2:] != expected:
                        raise ValueError(
                            f"`{name}` yields tiles of shape {probe.shape} but {expected} pixels were expected; pass "
                            "`rasterize_kwargs={'target_unit_to_pixels': ...}` so that tiles are the same size."
                        )
                    sources[name] = _Source(kind, None, channels, probe.dtype, fallback)
        return sources

    def _add_item(self, X: Any, obs: pd.DataFrame | None, var: pd.DataFrame | None, item: _Item) -> None:
        """Add a dataset together with its pixel sources, atomically."""
        signature = {name: src.signature for name, src in item.sources.items()}
        hw = (item.geometry.height, item.geometry.width)
        if self._items:
            if signature != self._signature:
                raise ValueError(
                    f"All datasets must provide the same modalities: expected {self._signature}, got {signature}."
                )
            if hw != self._tile_hw:
                raise ValueError(f"All datasets must have the same tile size: expected {self._tile_hw}, got {hw}.")
            if (item.coords is None) != (self._items[0].coords is None):
                raise ValueError("Either all or none of the datasets must have `obsm['spatial']`.")
        self._add_dataset_unchecked(X, obs, var)
        self._signature, self._tile_hw = signature, hw
        self._items.append(item)

    def add_sdatas(self, sdatas: Sequence[SpatialData], **kwargs: Any) -> Self:
        """Append several SpatialData objects, see :meth:`add_sdata`."""
        check_lt_1([len(sdatas)], ["Number of sdatas"])
        for sdata in sdatas:
            self.add_sdata(sdata, **kwargs)
        return self

    def use_collection(self, collection: SpatialDatasetCollection, **kwargs: Any) -> Self:  # type: ignore[override]
        """Load from a :class:`spatialbatch.SpatialDatasetCollection`. Can only be called once."""
        if collection.is_empty:
            raise ValueError("SpatialDatasetCollection is empty")
        if self._collection_added:
            raise RuntimeError("You should not add multiple collections; add them to one collection instead.")
        for group in collection:
            self._add_group(group)
        self._collection_added = True
        return self

    def _add_group(self, group: zarr.Group) -> None:
        from spatialbatch.io import open_dataset_group

        X, obs, var, item = open_dataset_group(group, self._image_names, self._label_names, self._annotations)
        self._add_item(X, obs, var, item)

    def _unsupported(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("A spatialbatch.Loader holds pixels too: use `add_sdata`, `add_sdatas` or `use_collection`.")

    add_adata = add_adatas = add_dataset = add_datasets = _unsupported  # type: ignore[assignment]

    @property
    def n_tiles_px(self) -> tuple[int, int]:
        """Height and width in pixels of every tile."""
        if self._tile_hw is None:
            raise ValueError("No datasets added yet")
        return self._tile_hw

    # -- reading -------------------------------------------------------------------------------------------------

    async def _index_pixels(self, dataset_index_to_rows: OrderedDict[int, np.ndarray]) -> dict[str, np.ndarray]:
        """Fetch the tiles of the requested rows of every modality into preallocated buffers.

        The buffers have the row order of the ``X`` buffer of :meth:`_index_datasets`: grouped by dataset.
        """
        assert self._signature is not None and self._tile_hw is not None
        total = sum(len(rows) for rows in dataset_index_to_rows.values())
        th, tw = self._tile_hw
        buffers = {
            name: np.empty((total, *(() if channels is None else (channels,)), th, tw), dtype)
            for name, (_, channels, dtype) in self._signature.items()
        }
        tasks, offset = [], 0
        for dataset_idx, rows in dataset_index_to_rows.items():
            item = self._items[dataset_idx]
            for name, source in item.sources.items():
                out = buffers[name][offset : offset + len(rows)]
                if source.fallback is None:
                    tasks.append(self._fetch_tiles(source, item.geometry, rows, out))
                else:
                    tasks.append(self._fetch_tiles_fallback(source, rows, out))
            offset += len(rows)
        await asyncio.gather(*tasks)
        return buffers

    async def _fetch_tiles(self, source: _Source, geometry: TileGeometry, rows: np.ndarray, out: np.ndarray) -> None:
        """Fill `out` with the tiles of `rows`, reading each window of the array that holds several tiles only once."""
        assert source.array is not None
        th, tw = geometry.height, geometry.width
        height, width = source.array.shape[-2:]
        y0, x0 = geometry.y0[rows] - source.origin[1], geometry.x0[rows] - source.origin[0]
        cy0, cy1 = np.clip(y0, 0, height), np.clip(y0 + th, 0, height)
        cx0, cx1 = np.clip(x0, 0, width), np.clip(x0 + tw, 0, width)
        inside = (cy1 > cy0) & (cx1 > cx0)
        whole = inside & (cy1 - cy0 == th) & (cx1 - cx0 == tw)
        if not whole.all():  # only the tiles at the border
            out[~whole] = self._pad_value
        members = np.flatnonzero(inside)
        if members.size == 0:
            return
        cells = (cy0[members] // _WINDOW_PX) * (width // _WINDOW_PX + 2) + cx0[members] // _WINDOW_PX
        order = np.argsort(cells, kind="stable")
        members, cells = members[order], cells[order]
        bounds = np.flatnonzero(np.diff(cells)) + 1
        arr = source.array._async_array

        async def read_window(group: np.ndarray) -> None:
            wy0, wy1 = int(cy0[group].min()), int(cy1[group].max())
            wx0, wx1 = int(cx0[group].min()), int(cx1[group].max())
            selection = (slice(wy0, wy1), slice(wx0, wx1))
            window = await arr.getitem(selection if source.channels is None else (slice(None), *selection))
            for j, sy0, sy1, sx0, sx1, dy0, dx0 in zip(
                group.tolist(),
                (cy0[group] - wy0).tolist(),
                (cy1[group] - wy0).tolist(),
                (cx0[group] - wx0).tolist(),
                (cx1[group] - wx0).tolist(),
                (cy0[group] - y0[group]).tolist(),
                (cx0[group] - x0[group]).tolist(),
                strict=True,
            ):
                out[j, ..., dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = window[..., sy0:sy1, sx0:sx1]

        await asyncio.gather(*(read_window(g) for g in np.split(members, bounds)))

    async def _fetch_tiles_fallback(self, source: _Source, rows: np.ndarray, out: np.ndarray) -> None:
        """Fill `out` with one spatialdata call per tile, spread over threads."""
        assert source.fallback is not None
        n_threads = zarr.config.get("threading.max_workers") or os.cpu_count() or 1

        def work() -> None:
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                for j, tile in enumerate(pool.map(source.fallback, rows.tolist())):
                    out[j] = tile

        await asyncio.get_running_loop().run_in_executor(None, work)

    async def _index_all(
        self, dataset_index_to_rows: OrderedDict[int, np.ndarray]
    ) -> tuple[Any, dict[str, np.ndarray]]:
        """Fetch ``X`` and all pixel modalities concurrently."""
        x, pixels = await asyncio.gather(
            self._index_datasets(dataset_index_to_rows), self._index_pixels(dataset_index_to_rows)
        )
        return x, pixels

    def _maybe_accumulate_coords(self, dataset_index_to_rows: OrderedDict[int, np.ndarray]) -> np.ndarray | None:
        coords = [self._items[idx].coords for idx in dataset_index_to_rows]
        if any(c is None for c in coords):
            return None
        return np.concatenate([c[rows] for c, rows in zip(coords, dataset_index_to_rows.values(), strict=True)])

    def _iter(self) -> Iterator[SpatialLoaderOutput]:  # type: ignore[override]
        check_lt_1([len(self._train_datasets), self.n_obs], ["Number of datasets", "Number of observations"])
        is_sparse = not issubclass(self.dataset_type, zarr.Array)
        positions = np.empty(0, dtype=np.intp)
        for load_request in self._batch_sampler.sample(self.n_obs):
            requests_to_load = load_request.get("requests", None)
            if requests_to_load is None:
                raise KeyError("load_request must contain either 'requests'.")
            splits = load_request["splits"]

            dataset_index_to_rows, order = self._requests_to_dataset_rows(requests_to_load)
            # See annbatch.Loader._iter: `inv` maps a position in the request to its position in the buffers.
            n = order.size
            inv_buffer = np.empty(n, dtype=np.intp)
            if n > positions.size:
                positions = np.arange(n, dtype=np.intp)
            inv = inv_buffer[:n]
            inv[order] = positions[:n]

            raw_out, pixels = zsync.sync(self._index_all(dataset_index_to_rows))

            if is_sparse:
                in_memory_data = self._sp_module.csr_matrix(
                    tuple(self._np_module.asarray(e) for e in raw_out.elems), shape=raw_out.shape, dtype=raw_out.dtype
                )
            else:
                in_memory_data = self._np_module.asarray(raw_out)

            obs = self._maybe_accumulate_obs(dataset_index_to_rows)
            index = self._maybe_accumulate_indices(dataset_index_to_rows)
            coords = self._maybe_accumulate_coords(dataset_index_to_rows)
            for split in splits:
                sel = inv[split]
                data = in_memory_data[sel]
                batch: SpatialLoaderOutput = {
                    "X": data if self._to is None else convert(data, False, self._to),
                    "obs": obs.iloc[sel] if obs is not None else None,
                    "var": self._var,
                    "index": index[sel] if index is not None else None,
                    "images": {name: self._convert_tiles(pixels[name][sel]) for name in self._image_names},
                    "labels": {name: self._convert_tiles(pixels[name][sel]) for name in self._label_names},
                    "coords": coords[sel] if coords is not None else None,
                }
                yield batch if self.transform is None else self.transform(batch)

    def _convert_tiles(self, tiles: np.ndarray) -> Any:
        return tiles if self._to is None else convert(tiles, False, self._to)
