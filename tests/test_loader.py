from __future__ import annotations

import numpy as np
import pytest
from annbatch.samplers import SequentialSampler

from spatialbatch import Loader
from spatialbatch.abc import Sampler
from spatialbatch.samplers import RandomSampler

TILE = 32
KW = {"region": "cells"}
TILES = {"tile_dim_in_units": TILE, "rasterize": True, "rasterize_kwargs": {"target_unit_to_pixels": 1.0}}


def _identity(batch):
    return batch


def _loader(**kwargs) -> Loader:
    return Loader(images=["img"], labels=["lab"], return_index=True, **{**TILES, **kwargs})


def test_tiles_are_the_pixels_under_each_cell(synthetic, ref_tile):
    sdata, raw = synthetic
    loader = _loader(batch_size=10, chunk_size=10, preload_nchunks=2).add_sdata(sdata, **KW)
    seen = 0
    for batch in loader:
        for row, idx in enumerate(batch["index"]):
            cx, cy = raw["centers"][idx]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(raw["image"], cx, cy))
            np.testing.assert_array_equal(batch["labels"]["lab"][row], ref_tile(raw["labels"], cx, cy))
            np.testing.assert_array_equal(batch["coords"][row], (raw["centers"][idx] / 2).astype(np.float32))
            assert batch["obs"]["label"].iloc[row] == idx + 1
            seen += 1
    assert seen == 200


def test_every_cell_is_visited_once_per_epoch(synthetic, make, tmp_path):
    sdata, _ = synthetic
    other, _ = make(tmp_path / "b.zarr", n_cells=150, seed=1)
    loader = _loader(
        batch_size=16, chunk_size=8, preload_nchunks=4, shuffle=True, rng=np.random.default_rng(0)
    ).add_sdatas([sdata, other], **KW)
    assert loader.n_obs == 350
    index = np.concatenate([b["index"] for b in loader])
    np.testing.assert_array_equal(np.sort(index), np.arange(350))
    assert not np.array_equal(index, np.arange(350))  # actually shuffled
    assert sum(len(b["index"]) for b in loader) == 350
    assert len(loader) == -(-350 // 16)


def test_rows_of_each_dataset_get_their_own_pixels(synthetic, make, ref_tile, tmp_path):
    sdata, raw_a = synthetic
    other, raw_b = make(tmp_path / "b.zarr", n_cells=150, seed=1, shape=(280, 300))
    loader = _loader(
        batch_size=25, chunk_size=25, preload_nchunks=2, shuffle=True, rng=np.random.default_rng(2)
    ).add_sdatas([sdata, other], **KW)
    for batch in loader:
        for row, idx in enumerate(batch["index"]):
            raw, local = (raw_a, idx) if idx < 200 else (raw_b, idx - 200)
            cx, cy = raw["centers"][local]
            np.testing.assert_array_equal(batch["images"]["img"][row], ref_tile(raw["image"], cx, cy))
            assert batch["obs"]["label"].iloc[row] == local + 1


def test_sampler_is_an_annbatch_sampler():
    sampler = RandomSampler(chunk_size=4, preload_nchunks=2, batch_size=4, rng=np.random.default_rng(0))
    assert isinstance(sampler, Sampler)
    flat = np.concatenate([np.arange(s.start, s.stop) for r in sampler.sample(40) for s in r["requests"]])
    np.testing.assert_array_equal(np.sort(flat), np.arange(40))


def test_custom_sampler_and_transform(synthetic):
    sdata, _ = synthetic
    sampler = SequentialSampler(chunk_size=10, preload_nchunks=2, batch_size=10, mask=slice(0, 40))
    loader = Loader(
        images=["img"], batch_sampler=sampler, transform=lambda b: b["images"]["img"].sum(), **TILES
    ).add_sdata(sdata, **KW)
    assert len(list(loader)) == 4


def test_torch_tensors(synthetic):
    torch = pytest.importorskip("torch")
    sdata, _ = synthetic
    batch = next(iter(_loader(batch_size=8, chunk_size=8, preload_nchunks=1, to="torch").add_sdata(sdata, **KW)))
    assert isinstance(batch["images"]["img"], torch.Tensor)
    assert tuple(batch["images"]["img"].shape) == (8, 1, TILE, TILE)


@pytest.mark.parametrize("num_workers", [0, 2])
def test_torch_dataloader(synthetic, num_workers):
    torch_data = pytest.importorskip("torch.utils.data")
    sdata, _ = synthetic
    loader = _loader(
        batch_size=10,
        chunk_size=10,
        preload_nchunks=2,
        shuffle=True,
        drop_last=True,
        rng=np.random.default_rng(0),
        to="torch",
    ).add_sdata(sdata, **KW)
    dataloader = torch_data.DataLoader(loader, batch_size=None, num_workers=num_workers, collate_fn=_identity)
    index = np.concatenate([np.asarray(b["index"]) for b in dataloader])
    assert len(np.unique(index)) == len(index) == 200


def test_errors(synthetic, make, tmp_path):
    sdata, _ = synthetic
    with pytest.raises(ValueError, match="Cannot specify"):
        Loader(batch_sampler=RandomSampler(4, 2, 4), batch_size=2)
    with pytest.raises(ValueError, match="At least one"):
        Loader(images=[])
    with pytest.raises(TypeError, match="add_sdata"):
        Loader().add_adata(None)
    with pytest.raises(ValueError, match="same integer size"):
        Loader(images=["img"], tile_scale=2.0).add_sdata(
            make(tmp_path / "v.zarr", radius=np.linspace(2, 6, 200))[0], **KW
        )
    with pytest.raises(ValueError, match="not valid for the region"):
        _loader().add_sdata(sdata, cs="nope", **KW)
    loader = _loader().add_sdata(sdata, **KW)
    three_channels, _ = make(tmp_path / "c.zarr", seed=3, channels=3)
    with pytest.raises(ValueError, match="same modalities"):
        loader.add_sdata(three_channels, **KW)
    assert loader.n_obs == 200  # the failed add left the loader untouched
