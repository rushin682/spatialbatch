"""Write a collection once, then mini-batch any subset of its elements (table, images, labels)."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from spatialbatch import Loader, SpatialDatasetCollection, center_crop, write_sharded_sdata
from spatialbatch.utils import DEFAULT_PIXEL_CHUNK, DEFAULT_TILE_PX

TILE = 32
KW = {"region": "cells"}
SMALL = {"block_px": 64, "image_chunk": 32, "image_shard": 128, "n_obs_per_chunk": 16, "shard_size": 64}
ORDERED = {"batch_size": 50, "chunk_size": 50, "preload_nchunks": 1, "return_index": True}


@pytest.fixture
def collection(synthetic, tmp_path):
    collection = SpatialDatasetCollection(tmp_path / "all.zarr")
    collection.add_sdatas([tmp_path / "a.zarr"], images=["img"], labels=["lab"], tile_dim_in_units=TILE, **SMALL, **KW)
    return collection


def _by_index(loader: Loader) -> dict[int, dict]:
    """Every row of one epoch, keyed by its position in the loader."""
    rows: dict[int, dict] = {}
    for batch in loader:
        x = batch["X"]
        for j, idx in enumerate(batch["index"]):
            rows[int(idx)] = {
                "X": None if x is None else x[j].toarray().ravel(),
                "img": batch["images"]["img"][j] if "img" in batch["images"] else None,
                "lab": batch["labels"]["lab"][j] if "lab" in batch["labels"] else None,
                "id": batch["obs"]["instance_id"].iloc[j] if batch["obs"] is not None else None,
            }
    return rows


def test_one_collection_loaded_four_ways(collection):
    full = _by_index(Loader(images=["img"], labels=["lab"], **ORDERED).use_collection(collection))
    table = _by_index(Loader(images=(), labels=(), **ORDERED).use_collection(collection))
    images = _by_index(Loader(table=False, images=["img"], **ORDERED).use_collection(collection))
    labels = _by_index(Loader(table=False, images=(), labels=["lab"], **ORDERED).use_collection(collection))
    assert len(full) == len(table) == len(images) == len(labels) == 200
    for idx, row in full.items():
        np.testing.assert_array_equal(table[idx]["X"], row["X"])
        assert table[idx]["img"] is None and table[idx]["lab"] is None
        assert images[idx]["X"] is None and images[idx]["lab"] is None
        np.testing.assert_array_equal(images[idx]["img"], row["img"])
        assert labels[idx]["X"] is None and labels[idx]["img"] is None
        np.testing.assert_array_equal(labels[idx]["lab"], row["lab"])
        assert table[idx]["id"] == images[idx]["id"] == row["id"]


def test_unrequested_elements_are_never_read(collection, monkeypatch):
    calls = {"X": 0, "tiles": 0}

    async def no_tiles(*args, **kwargs):
        calls["tiles"] += 1

    original = Loader._index_datasets

    async def count_x(self, *args, **kwargs):
        calls["X"] += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(Loader, "_fetch_tiles", no_tiles)
    monkeypatch.setattr(Loader, "_index_datasets", count_x)
    batch = next(iter(Loader(images=(), batch_size=8).use_collection(collection)))
    assert calls == {"X": 1, "tiles": 0}
    assert batch["images"] == {} and batch["labels"] == {} and batch["X"].shape == (8, 20)

    calls.update(X=0, tiles=0)
    batch = next(iter(Loader(table=False, images=["img"], batch_size=8).use_collection(collection)))
    assert calls["X"] == 0 and calls["tiles"] > 0
    assert batch["X"] is None and batch["var"] is None


def test_table_only_in_place(synthetic):
    sdata, _ = synthetic
    loader = Loader(images=(), **ORDERED).add_sdata(sdata, **KW)
    with pytest.raises(ValueError, match="no tiles"):
        _ = loader.n_tiles_px
    rows = _by_index(loader)
    expected = sdata["table"].X.toarray()
    for idx, row in rows.items():
        np.testing.assert_array_equal(row["X"], expected[idx])


def test_table_only_collection(synthetic, tmp_path):
    collection = SpatialDatasetCollection(tmp_path / "table.zarr")
    collection.add_sdatas([tmp_path / "a.zarr"], images=(), labels=(), **SMALL, **KW)
    group = next(iter(collection))
    assert "images" not in group and "tile_origin" not in group["obsm"]
    assert len(_by_index(Loader(images=(), **ORDERED).use_collection(collection))) == 200
    with pytest.raises(ValueError, match="not in the collection"):
        Loader(images=["img"]).use_collection(collection)


def test_load_table_and_write_x(synthetic, tmp_path):
    sdata, _ = synthetic

    def keep_ids_as_float64(adata):
        adata = adata.copy()
        adata.obs = adata.obs[["instance_id"]]
        adata.X = adata.X.astype(np.float64)
        return adata

    group = zarr.open_group(tmp_path / "c.zarr", mode="w")
    write_sharded_sdata(
        group, sdata, key="dataset_0", images=["img"], load_table=keep_ids_as_float64, tile_dim_in_units=TILE, **KW
    )
    assert list(group["dataset_0"]["obs"].attrs["column-order"]) == ["instance_id"]
    assert group["dataset_0"]["X"]["data"].dtype == np.float64

    with pytest.raises(ValueError, match="keep the rows"):
        write_sharded_sdata(group, sdata, key="dataset_1", images=["img"], load_table=lambda a: a[:10], **KW)

    write_sharded_sdata(group, sdata, key="dataset_2", images=["img"], write_X=False, tile_dim_in_units=TILE, **KW)
    assert group["dataset_2"]["X"].attrs["shape"] == [200, 0]


def test_defaults_are_1024_pixel_tiles_in_128_pixel_chunks(synthetic, tmp_path):
    assert DEFAULT_TILE_PX == 1024 and DEFAULT_PIXEL_CHUNK == 128 == DEFAULT_TILE_PX // 8
    sdata, _ = synthetic  # a 300x340 image: chunks are clipped to it
    collection = SpatialDatasetCollection(tmp_path / "d.zarr")
    collection.add_sdatas([tmp_path / "a.zarr"], images=["img"], **KW)
    group = next(iter(collection))
    assert group["images"]["img"].chunks == (1, 128, 128)
    assert Loader(images=["img"]).use_collection(collection).n_tiles_px == (1024, 1024)
    with pytest.warns(UserWarning, match="does not divide"):
        write_sharded_sdata(
            zarr.open_group(tmp_path / "e.zarr", mode="w"),
            sdata,
            key="dataset_0",
            images=["img"],
            image_chunk=96,
            image_shard=960,
            **KW,
        )


@pytest.mark.parametrize("size", [8, (8, 16)])
def test_center_crop_keeps_the_cell_centered(collection, size):
    loader = Loader(images=["img"], labels=["lab"], batch_size=8, transform=lambda b: center_crop(b, size))
    batch = next(iter(loader.use_collection(collection)))
    full = next(iter(Loader(images=["img"], labels=["lab"], batch_size=8).use_collection(collection)))
    h, w = (size, size) if isinstance(size, int) else size
    y0, x0 = (TILE - h) // 2, (TILE - w) // 2
    assert batch["images"]["img"].shape == (8, 1, h, w) and batch["labels"]["lab"].shape == (8, h, w)
    np.testing.assert_array_equal(batch["images"]["img"], full["images"]["img"][..., y0 : y0 + h, x0 : x0 + w])
    with pytest.raises(ValueError, match="Cannot crop"):
        center_crop(full, TILE + 1)
