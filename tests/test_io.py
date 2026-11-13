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


# -- collections written from a table whose obsm["spatial"] holds the cell centres in pixels -------------------------

SHAPE = (300, 340)
ARRAYS = {"block_px": 64, "image_chunk": 32, "image_shard": 128, "n_obs_per_chunk": 16, "shard_size": 64}


def _centred_table(n_cells: int = 200, seed: int = 0, shape: tuple[int, int] = SHAPE):
    import anndata as ad
    import pandas as pd
    from scipy import sparse as sp

    rng = np.random.default_rng(seed)
    height, width = shape
    image = rng.integers(1, 60000, (2, height, width), dtype=np.uint16)
    labels = rng.integers(0, 50, (height, width)).astype(np.uint32)
    centres = rng.uniform([0, 0], [width, height], (n_cells, 2))  # pixels; some crops cross the border
    ids = [f"cell-{i}" for i in range(n_cells)]
    table = ad.AnnData(
        sp.random(n_cells, 20, density=0.3, format="csr", dtype=np.float32, random_state=seed),
        obs=pd.DataFrame({"cell": np.arange(n_cells)}, index=ids),
    )
    table.obsm["spatial"] = centres
    return table, image, labels


def _write(path, table, image, labels, **kwargs):
    from spatialbatch import write_collection_dataset

    group = zarr.open_group(path, mode="a")
    write_collection_dataset(
        group, key="dataset_0", table=table, images={"img": image}, labels={"lab": labels}, **ARRAYS, **kwargs
    )
    return SpatialDatasetCollection(path)


def test_write_collection_dataset_roundtrip(tmp_path):
    table, image, labels = _centred_table()
    collection = _write(tmp_path / "c.zarr", table, image, labels, attrs={"source": "synthetic"})
    group = next(iter(collection))
    np.testing.assert_array_equal(group["images"]["img"][...], image)
    np.testing.assert_array_equal(group["labels"]["lab"][...], labels)
    assert group["images"]["img"].shards == (2, 128, 128) and group["images"]["img"].chunks == (2, 32, 32)
    meta = group.attrs["spatialbatch"]
    assert meta["encoding-version"] == "0.2.0" and meta["centres"] == "obsm/spatial"
    assert meta["images"] == ["img"] and meta["labels"] == ["lab"] and meta["block_px"] == 64
    assert meta["source"] == "synthetic" and "tile_hw" not in meta and "tile_origin" not in group["obsm"]
    spatial = np.asarray(group["obsm"]["spatial"][...])
    np.testing.assert_array_equal(morton_order(spatial[:, 0], spatial[:, 1], 64), np.arange(200))
    cells = np.asarray(group["obs"]["cell"][...])
    np.testing.assert_array_equal(spatial, table.obsm["spatial"][cells])
    assert not np.array_equal(cells, np.arange(200))  # actually reordered


def test_array_smaller_than_shard_with_default_chunks(tmp_path):
    from spatialbatch import write_collection_dataset

    table, image, labels = _centred_table()  # 300 x 340: not a multiple of the default 256-px chunk
    group = zarr.open_group(tmp_path / "c.zarr", mode="a")
    write_collection_dataset(group, key="dataset_0", table=table, images={"img": image}, labels={"lab": labels})
    array = group["dataset_0"]["images"]["img"]
    assert array.chunks == (2, 256, 256) and array.shards == (2, 512, 512)
    np.testing.assert_array_equal(array[...], image)


@pytest.mark.parametrize("crop_px", [16, 33])
def test_crop_at_read_matches_reference(tmp_path, ref_tile, crop_px):
    table, image, labels = _centred_table()
    collection = _write(tmp_path / "c.zarr", table, image, labels)
    loader = Loader(
        images=["img"], labels=["lab"], crop_px=crop_px, return_index=True, batch_size=20, chunk_size=10
    ).use_collection(collection)
    assert loader.n_tiles_px == (crop_px, crop_px)
    seen = []
    for batch in loader:
        for row, cell in enumerate(batch["obs"]["cell"]):
            cx, cy = table.obsm["spatial"][cell]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(image, cx, cy, crop_px))
            np.testing.assert_array_equal(batch["labels"]["lab"][row], ref_tile(labels, cx, cy, crop_px))
            np.testing.assert_array_equal(batch["coords"][row], (cx, cy))
            seen.append(cell)
    np.testing.assert_array_equal(np.sort(seen), np.arange(200))
    with pytest.raises(ValueError, match="crop_px"):
        Loader(images=["img"]).use_collection(collection)


def test_dask_input_not_computed_whole(tmp_path, monkeypatch):
    da = pytest.importorskip("dask.array")
    table, image, labels = _centred_table()
    lazy_image, lazy_labels = da.from_array(image, chunks=(1, 100, 90)), da.from_array(labels, chunks=(70, 110))

    def refuse(*args, **kwargs):
        raise AssertionError("the whole dask array was computed")

    monkeypatch.setattr(da.Array, "compute", refuse)
    monkeypatch.setattr(da.Array, "__array__", refuse)
    group = next(iter(_write(tmp_path / "c.zarr", table, lazy_image, lazy_labels)))
    monkeypatch.undo()
    np.testing.assert_array_equal(group["images"]["img"][...], image)
    np.testing.assert_array_equal(group["labels"]["lab"][...], labels)


@pytest.mark.parametrize("kind", ["scipy", "dask", "dask-dense", "backed"])
def test_written_x_is_the_input_x_in_order(tmp_path, kind):
    import anndata as ad

    table, image, labels = _centred_table()
    expected = table.X.copy()
    table.uns["spatialdata_attrs"] = {"region": "cells"}
    if kind == "dask":
        da = pytest.importorskip("dask.array")
        table.X = da.from_array(expected, chunks=(64, 20))
    elif kind == "dask-dense":
        da = pytest.importorskip("dask.array")
        table.X = da.from_array(expected.toarray(), chunks=(64, 7))
    elif kind == "backed":
        ad.io.write_elem(zarr.open_group(tmp_path / "x.zarr", mode="w"), "X", expected)
        backed = ad.io.sparse_dataset(zarr.open_group(tmp_path / "x.zarr", mode="r")["X"])
        table.X = backed  # read in order from the store, not loaded first
    group = next(iter(_write(tmp_path / "c.zarr", table, image, labels)))
    cells = np.asarray(group["obs"]["cell"][...])
    assert not np.array_equal(cells, np.arange(table.n_obs))
    written = ad.io.read_elem(group["X"])
    np.testing.assert_array_equal(
        np.asarray(written.todense() if hasattr(written, "todense") else written), expected[cells].toarray()
    )
    assert "spatialdata_attrs" in table.uns and "spatialdata_attrs" not in group["uns"]


def test_sorted_table_is_not_copied(monkeypatch, tmp_path):
    import spatialbatch.io as sio

    table, image, labels = _centred_table()
    order = morton_order(table.obsm["spatial"][:, 0], table.obsm["spatial"][:, 1], 64)
    table = table[order].copy()
    seen = {}
    real = sio._write_dataset
    monkeypatch.setattr(
        sio, "_write_dataset", lambda group, adata, **kw: seen.update(X=adata.X) or real(group, adata, **kw)
    )
    _write(tmp_path / "c.zarr", table, image, labels)
    assert seen["X"] is table.X


def test_old_encoding_still_loads(collection, ref_tile):
    collection, raws = collection
    assert all(g.attrs["spatialbatch"]["encoding-version"] == "0.1.0" for g in collection)
    with pytest.warns(UserWarning, match="crop_px"):
        loader = Loader(images=["img"], labels=["lab"], crop_px=8, batch_size=50, chunk_size=50).use_collection(
            collection
        )
    assert loader.n_tiles_px == (TILE, TILE)  # tile_hw rules
    seen = 0
    for batch in loader:
        for row, instance in enumerate(batch["obs"]["instance_id"]):
            seed, cell = (int(p) for p in str(instance).removeprefix("cell").split("-"))
            cx, cy = raws[seed]["centers"][cell]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(raws[seed]["image"], cx, cy))
            np.testing.assert_array_equal(batch["labels"]["lab"][row], ref_tile(raws[seed]["labels"], cx, cy))
            seen += 1
    assert seen == 350


def test_sddb_contract_is_checked(tmp_path, monkeypatch):
    import sys
    import types

    table, image, labels = _centred_table(n_cells=40)
    table.uns["sddb_contract"] = {"version": "0.1.0"}
    collection = _write(tmp_path / "c.zarr", table, image, labels)
    kwargs = {"images": ["img"], "crop_px": 16}
    monkeypatch.setitem(sys.modules, "sddb", None)  # sddb not installed: only the version is checked
    with pytest.warns(UserWarning, match="sddb"):
        assert Loader(**kwargs).use_collection(collection).n_obs == 40
    bad, *_ = _centred_table(n_cells=40)
    bad.uns["sddb_contract"] = {"version": "zero"}
    with pytest.raises(ValueError, match="not semver"):
        Loader(**kwargs).use_collection(_write(tmp_path / "bad.zarr", bad, image, labels))
    contract = types.ModuleType("sddb.contract")
    contract.validate_dataset = lambda group: ["obsm/spatial is outside the image"]
    monkeypatch.setitem(sys.modules, "sddb", types.ModuleType("sddb"))
    monkeypatch.setitem(sys.modules, "sddb.contract", contract)
    with pytest.raises(ValueError, match="outside the image"):
        Loader(**kwargs).use_collection(collection)
    contract.validate_dataset = lambda group: []
    assert Loader(**kwargs).use_collection(collection).n_obs == 40


def test_raw_is_written_in_order(tmp_path):
    import anndata as ad

    table, image, labels = _centred_table()
    table.raw = table.copy()
    expected = table.X.toarray()
    group = next(iter(_write(tmp_path / "c.zarr", table, image, labels)))
    cells = np.asarray(group["obs"]["cell"][...])
    np.testing.assert_array_equal(ad.io.read_elem(group["raw"]["X"]).toarray(), expected[cells])
