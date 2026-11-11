from __future__ import annotations

import numpy as np
import pytest
import zarr

from spatialbatch import Loader, SpatialDatasetCollection
from spatialbatch.utils import morton_order

TILE = 32
TILES = {"tile_dim_in_units": TILE}
KW = {"region": "cells"}


@pytest.fixture
def collection(synthetic, make, tmp_path):
    a = synthetic
    b = make(tmp_path / "b.zarr", n_cells=150, seed=1, shape=(280, 300))
    collection = SpatialDatasetCollection(tmp_path / "collection.zarr")
    collection.add_sdatas(
        [tmp_path / "a.zarr", tmp_path / "b.zarr"],
        images=["img"],
        labels=["lab"],
        block_px=64,
        image_chunk=32,
        image_shard=128,
        n_obs_per_chunk=16,
        shard_size=64,
        **KW,
        **TILES,
    )
    return collection, [a[1], b[1]]


def test_collection_layout(collection):
    collection, _ = collection
    assert not collection.is_empty
    assert [g.name.strip("/") for g in collection] == ["dataset_0", "dataset_1"]
    group = next(iter(collection))
    assert group["images"]["img"].shards == (1, 128, 128) and group["images"]["img"].chunks == (1, 32, 32)
    assert group["labels"]["lab"].shards == (128, 128)
    assert group["obsm"]["tile_origin"].shape == (200, 2)
    assert len(collection.obs()) == 350


def test_rows_are_sorted_along_a_z_curve(collection):
    collection, raws = collection
    group = next(iter(collection))
    spatial = np.asarray(group["obsm"]["spatial"][...]) * 2
    np.testing.assert_array_equal(morton_order(spatial[:, 0], spatial[:, 1], 64), np.arange(200))
    random_gap = np.linalg.norm(np.diff(raws[0]["centers"], axis=0), axis=1).mean()
    assert np.linalg.norm(np.diff(spatial, axis=0), axis=1).mean() < 0.7 * random_gap


def test_loader_from_collection_reads_the_right_tiles(collection, ref_tile):
    collection, raws = collection
    loader = Loader(
        images=["img"],
        labels=["lab"],
        return_index=True,
        batch_size=20,
        chunk_size=10,
        preload_nchunks=4,
        shuffle=True,
        rng=np.random.default_rng(0),
    ).use_collection(collection)
    assert loader.n_obs == 350
    seen = []
    for batch in loader:
        for row, idx in enumerate(batch["index"]):
            raw = raws[0] if idx < 200 else raws[1]
            cell = int(
                str(batch["obs"]["instance_id"].iloc[row]).split("-")[1]
            )  # rows are reordered: find the cell by id
            cx, cy = raw["centers"][cell]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(raw["image"], cx, cy))
            np.testing.assert_array_equal(batch["labels"]["lab"][row], ref_tile(raw["labels"], cx, cy))
            seen.append(idx)
    np.testing.assert_array_equal(np.sort(seen), np.arange(350))


def test_collection_matches_in_place_loading(collection, synthetic):
    collection, _ = collection
    sdata, _ = synthetic
    kwargs = {"images": ["img"], "return_index": True, "batch_size": 50, "chunk_size": 50, "preload_nchunks": 1}
    in_place = Loader(**kwargs, **TILES, rasterize=True, rasterize_kwargs={"target_unit_to_pixels": 1.0}).add_sdata(
        sdata, **KW
    )
    by_id = {}
    for batch in in_place:
        for row in range(len(batch["index"])):
            by_id[batch["obs"]["instance_id"].iloc[row]] = batch["images"]["img"][row]
    from_collection = Loader(**kwargs, tile_dim_in_units=TILE).use_collection(collection)
    compared = 0
    for batch in from_collection:
        for row in range(len(batch["index"])):
            key = batch["obs"]["instance_id"].iloc[row]
            if key in by_id:  # the cells of the first store, which is the one loaded in place
                np.testing.assert_array_equal(batch["images"]["img"][row], by_id[key])
                compared += 1
    assert compared == 200


def test_collection_errors(synthetic, tmp_path):
    sdata, _ = synthetic
    group = zarr.open_group(tmp_path / "x.zarr", mode="w")
    from spatialbatch import write_sharded_sdata

    with pytest.raises(ValueError, match="divisible"):
        write_sharded_sdata(
            group, sdata, key="dataset_0", images=["img"], image_chunk=48, image_shard=100, **KW, **TILES
        )
    with pytest.raises(TypeError, match="add_sdatas"):
        SpatialDatasetCollection(tmp_path / "y.zarr").add_adatas([])
    with pytest.raises(ValueError, match="empty"):
        Loader(images=["img"]).use_collection(SpatialDatasetCollection(tmp_path / "empty.zarr"))
