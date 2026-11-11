"""spatialbatch must return exactly what spatialstream's ``XeniumDataset`` returns, on the real stores."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import zarr.core.sync as zsync
from annbatch.samplers import SequentialSampler

from spatialbatch import Loader
from spatialbatch.utils import to_spatialstream_batch

pytestmark = pytest.mark.slow

XeniumDataset = pytest.importorskip("spatialstream.dataloaders.xenium").XeniumDataset
RASTERIZE_KWARGS = {"target_unit_to_pixels": 1.0}
BORDER_CELLS = slice(5200, 5260)  # cells 5222-5224 of the first store cross the left border of the image


def _original(sdata, rasterize: bool) -> XeniumDataset:
    return XeniumDataset(
        sdata=sdata,
        regions_to_images={"cell_circles": "morphology_focus"},
        regions_to_coordinate_systems={"cell_circles": "global"},
        tile_scale=2.0,
        tile_dim_in_units=224,
        rasterize=rasterize,
        return_annotations=["cell_labels"],
        table_name="table",
        rasterize_kwargs=RASTERIZE_KWARGS if rasterize else {},
    )


def _loader(sdata, *, rasterize: bool, sampler=None, labels=(), **kwargs) -> Loader:
    return Loader(
        images=["morphology_focus"],
        labels=labels,
        tile_dim_in_units=224,
        rasterize=rasterize,
        rasterize_kwargs=RASTERIZE_KWARGS if rasterize else {},
        return_annotations=["cell_labels"],
        return_index=True,
        batch_sampler=sampler,
        **kwargs,
    ).add_sdata(sdata)


def _compare(sdata, rasterize: bool, loader: Loader, *, max_batches: int, skip_cropped: bool = False) -> int:
    original, seen = _original(sdata, rasterize), 0
    for i, batch in enumerate(loader):
        if i == max_batches:
            break
        mine = to_spatialstream_batch(batch, image="morphology_focus", annotations=["cell_labels"])
        for row, idx in enumerate(batch["index"]):
            ref = original[int(idx)]
            ref_image = ref["image"].values
            if skip_cropped and ref_image.shape != mine["image"][row].shape:
                continue  # rasterize=False crops tiles at the border, which cannot be batched; we pad instead
            np.testing.assert_array_equal(mine["image"][row], ref_image, err_msg=f"image of cell {idx}")
            np.testing.assert_array_equal(mine["gexp"][row], ref["gexp"], err_msg=f"gexp of cell {idx}")
            np.testing.assert_array_equal(mine["coord"][row], ref["coord"], err_msg=f"coord of cell {idx}")
            np.testing.assert_array_equal(
                mine["annotations"][row], ref["annotations"], err_msg=f"annotations of cell {idx}"
            )
            seen += 1
    return seen


@pytest.mark.parametrize("rasterize", [True, False])
def test_random_cells_match_xenium_dataset(sdata, rasterize):
    loader = _loader(
        sdata,
        rasterize=rasterize,
        shuffle=True,
        rng=np.random.default_rng(0),
        batch_size=32,
        chunk_size=32,
        preload_nchunks=4,
    )
    assert _compare(sdata, rasterize, loader, max_batches=4, skip_cropped=not rasterize) >= 100


def _stored_tile(s0, x0: int, y0: int, size: int) -> np.ndarray:
    """Independent reference: the stored pixels under a tile, zero-padded where it leaves the array."""
    height, width = s0.shape[1:]
    out = np.zeros((s0.shape[0], size, size), dtype=s0.dtype)
    cy0, cy1, cx0, cx1 = max(y0, 0), min(y0 + size, height), max(x0, 0), min(x0 + size, width)
    out[:, cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = s0.isel(y=slice(cy0, cy1), x=slice(cx0, cx1)).values
    return out


def test_border_cells_are_padded_like_rasterize(sdata, store_path):
    """Tiles crossing the border equal the stored pixels (zero-padded), and `XeniumDataset` except on the edge line.

    ``spatialdata.rasterize`` interpolates against the padding on the outermost pixel line of the array, where it returns
    0 for stored values of 1. The stored pixels are the ground truth, so that line is excluded from the comparison with
    ``XeniumDataset`` but the whole tile is compared with the stored pixels.
    """
    if not store_path.name.startswith("6T1a"):
        pytest.skip("border cells were located in the first store")
    s0 = XeniumDataset._ensure_single_scale(sdata["morphology_focus"])
    height, width = s0.shape[1:]
    sampler = SequentialSampler(chunk_size=20, preload_nchunks=3, batch_size=20, mask=BORDER_CELLS)
    loader = _loader(sdata, rasterize=True, sampler=sampler)
    geometry, original, n_border = loader._items[0].geometry, _original(sdata, True), 0
    for batch in loader:
        for tile, idx in zip(batch["images"]["morphology_focus"], batch["index"], strict=True):
            x0, y0 = int(geometry.x0[idx]), int(geometry.y0[idx])
            np.testing.assert_array_equal(tile, _stored_tile(s0, x0, y0, 224), err_msg=f"stored pixels of cell {idx}")
            xs, ys = x0 + np.arange(224)[None, :], y0 + np.arange(224)[:, None]
            edge_line = np.broadcast_to((xs == 0) | (xs == width - 1) | (ys == 0) | (ys == height - 1), (224, 224))
            ref = original[int(idx)]["image"].values
            np.testing.assert_array_equal(tile[:, ~edge_line], ref[:, ~edge_line], err_msg=f"cell {idx}")
            n_border += int(x0 < 0 or y0 < 0 or x0 + 224 > width or y0 + 224 > height)
    assert n_border >= 3  # the tiles under test do cross the border


def test_labels_match_spatialdata(sdata):
    from spatialdata import rasterize

    loader = _loader(
        sdata,
        rasterize=True,
        labels=["cell_labels"],
        shuffle=True,
        rng=np.random.default_rng(1),
        batch_size=16,
        chunk_size=16,
        preload_nchunks=2,
    )
    batch = next(iter(loader))
    tiles = loader._items[0].geometry
    for row, idx in enumerate(batch["index"][:8]):
        x0, y0 = int(tiles.x0[idx]), int(tiles.y0[idx])
        ref = rasterize(
            sdata["cell_labels"],
            axes=("x", "y"),
            min_coordinate=np.array([x0, y0]),
            max_coordinate=np.array([x0 + 224, y0 + 224]),
            target_coordinate_system="global",
            target_unit_to_pixels=1.0,
        )
        ref = ref["scale0"].ds["image"].values if hasattr(ref, "keys") else ref.values
        ref = ref[0] if ref.ndim == 3 else ref  # rasterize returns labels with a singleton channel axis
        np.testing.assert_array_equal(batch["labels"]["cell_labels"][row], ref)
        centre = batch["labels"]["cell_labels"][row, 112, 112]
        assert centre in (0, batch["obs"]["cell_labels"].iloc[row])  # the cell sits in the middle of its own tile


COLLECTION = Path(__file__).parents[1] / "data" / "collection.zarr"
DATA_DIR = Path(os.environ.get("SPATIALBATCH_DATA", Path(__file__).parents[2] / "sddb-downloads"))
STORES = ["6T1aLES2sndpL1bQ0001.zarr", "1LYWTr2DRrkM6NTY0001.zarr", "ftcwfWqbDwNk5EMp0001.zarr"]


@pytest.mark.skipif(not COLLECTION.exists(), reason="build it with benchmarks/build_collection.py")
def test_collection_matches_in_place_on_real_stores():
    """The sorted, sharded copy of every store yields the very tiles that are read in place from the original."""
    import spatialdata as sd

    from spatialbatch import SpatialDatasetCollection

    kwargs = {
        "images": ["morphology_focus"],
        "labels": ["cell_labels"],
        "return_annotations": ["cell_labels"],
        "return_index": True,
        "batch_size": 32,
        "chunk_size": 32,
        "preload_nchunks": 4,
        "shuffle": True,
    }
    from_collection = Loader(**kwargs, tile_dim_in_units=224, rng=np.random.default_rng(0)).use_collection(
        SpatialDatasetCollection(COLLECTION, mode="r")
    )
    in_place = [
        Loader(
            images=["morphology_focus"],
            labels=["cell_labels"],
            tile_dim_in_units=224,
            rasterize=True,
            rasterize_kwargs=RASTERIZE_KWARGS,
            return_annotations=["cell_labels"],
            return_index=True,
            batch_size=1,
            chunk_size=1,
            preload_nchunks=1,
        ).add_sdata(sd.read_zarr(DATA_DIR / s))
        for s in STORES
    ]
    assert from_collection.n_obs == sum(loader.n_obs for loader in in_place)
    # cell ids are unique per store: `cell_labels` is the (1-based) position of the cell in the original table
    n_cells = [loader.n_obs for loader in in_place]
    slide_of = {}  # collection dataset -> store, found through the source recorded at write time
    for i, group in enumerate(SpatialDatasetCollection(COLLECTION, mode="r")):
        slide_of[i] = STORES.index(Path(group.attrs["spatialbatch"]["source"]).name)
    offsets = np.concatenate([[0], np.cumsum([from_collection._shapes[i][0] for i in range(len(STORES))])])
    checked = 0
    for batch in from_collection:
        for row, idx in enumerate(batch["index"]):
            dataset = int(np.searchsorted(offsets, idx, side="right") - 1)
            original_row = int(batch["obs"]["cell_labels"].iloc[row]) - 1  # position in the original store
            assert 0 <= original_row < n_cells[slide_of[dataset]]
            reference = in_place[slide_of[dataset]]
            tile = reference._items[0]
            out = np.empty((1, 1, 224, 224), np.uint16)
            zsync.sync(
                reference._fetch_tiles(tile.sources["morphology_focus"], tile.geometry, np.array([original_row]), out)
            )
            np.testing.assert_array_equal(batch["images"]["morphology_focus"][row], out[0])
            checked += 1
        if checked >= 600:
            break
    assert checked >= 600
