"""Elements whose pixel grid is not the coordinate system are resampled into it when a collection is written."""

from __future__ import annotations

import numpy as np
import pytest
import zarr

from spatialbatch import Loader, SpatialDatasetCollection
from spatialbatch.utils import aligned_tiles, tile_starts

KW = {"region": "cells"}
TILE_UNITS = 32
EXTRA_IMAGES = ["rot", "shift", "frac"]


def _rotation(angle_deg: float, scale: float, shear: float, tx: float, ty: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    linear = scale * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) @ np.array([[1, shear], [0, 1]])
    return np.array([[*linear[0], tx], [*linear[1], ty], [0, 0, 1]])


@pytest.fixture
def transformed(make, tmp_path):
    """The synthetic store plus elements that are rotated, shifted by whole pixels and shifted by a fraction."""
    import spatialdata as sd
    from spatialdata.models import Image2DModel, Labels2DModel
    from spatialdata.transformations import Affine, Translation

    path = tmp_path / "t.zarr"
    sdata, raw = make(path, n_cells=120, seed=3, channels=2)
    rng = np.random.default_rng(7)
    rotated = _rotation(30, 1.3, 0.05, 150, -60)
    affine = Affine(rotated, input_axes=("x", "y"), output_axes=("x", "y"))
    sdata["rot"] = Image2DModel.parse(
        rng.integers(1, 250, (3, 230, 260), dtype=np.uint8),
        dims=("c", "y", "x"),
        scale_factors=[2, 2],
        transformations={"global": affine},
    )
    sdata["rot_lab"] = Labels2DModel.parse(
        rng.integers(0, 1000, (230, 260)).astype(np.uint32),
        dims=("y", "x"),
        scale_factors=[2, 2],
        transformations={"global": affine},
    )
    sdata["shift"] = Image2DModel.parse(
        raw["image"], dims=("c", "y", "x"), transformations={"global": Translation([5, -7], axes=("x", "y"))}
    )
    sdata["frac"] = Image2DModel.parse(
        raw["image"], dims=("c", "y", "x"), transformations={"global": Translation([2.3, 0.6], axes=("x", "y"))}
    )
    for name in ("rot", "rot_lab", "shift", "frac"):
        sdata.write_element(name)
    return sd.read_zarr(path)


def _write(sdata, tmp_path, k: float) -> SpatialDatasetCollection:
    collection = SpatialDatasetCollection(tmp_path / f"collection_{k}.zarr")
    collection.add_sdatas(
        [sdata.path],
        images=["img", *EXTRA_IMAGES],
        labels=["lab", "rot_lab"],
        tile_dim_in_units=TILE_UNITS,
        target_unit_to_pixels=k,
        block_px=64,
        image_chunk=16,
        image_shard=64,
        n_obs_per_chunk=16,
        shard_size=64,
        **KW,
    )
    return collection


def _rasterized(element, x0: int, y0: int, size: int, k: float) -> np.ndarray:
    """Independent reference: ``spatialdata.rasterize`` of the tile box that starts on the target grid."""
    from spatialdata import rasterize

    tile = rasterize(
        element,
        axes=("x", "y"),
        min_coordinate=np.array([x0, y0]) / k,
        max_coordinate=np.array([x0 + size, y0 + size]) / k,
        target_coordinate_system="global",
        target_unit_to_pixels=k,
    )
    tile = next(iter(tile["scale0"].ds.values())) if hasattr(tile, "keys") else tile
    values = np.asarray(tile.values)
    is_labels = element.ndim == 2 if not hasattr(element, "keys") else len(element["scale0"].dims) == 2
    return values.reshape(values.shape[-2:]) if is_labels else values  # labels come with a singleton channel axis


@pytest.mark.parametrize("k", [1.0, 0.5, 0.25])
def test_every_element_matches_rasterize_on_the_target_grid(transformed, tmp_path, k):
    """At every resolution, each tile of each element is what ``rasterize`` returns for that tile's box.

    ``k=0.25`` makes ``rasterize`` read a coarser pyramid level of ``rot``, so the level choice is checked too.
    """
    collection = _write(transformed, tmp_path, k)
    _, tiles = aligned_tiles(transformed, "cells", "table", "global", tile_dim_in_units=TILE_UNITS)
    geometry = tile_starts(tiles, k)
    position = {cell: i for i, cell in enumerate(tiles.index)}
    size = geometry.width
    assert size == int(TILE_UNITS * k)

    loader = Loader(
        images=["img", *EXTRA_IMAGES],
        labels=["lab", "rot_lab"],
        batch_size=20,
        chunk_size=20,
        preload_nchunks=2,
        shuffle=True,
        rng=np.random.default_rng(0),
    ).use_collection(collection)
    compared = covered = 0
    for batch in loader:
        covered += int(((batch["images"]["rot"] > 0).mean(axis=(1, 2, 3)) > 0.5).sum())  # tiles mostly on `rot`
        for row, cell in enumerate(batch["obs"]["instance_id"]):
            i = position[cell]
            x0, y0 = int(geometry.x0[i]), int(geometry.y0[i])
            for kind, names in (("images", ["img", *EXTRA_IMAGES]), ("labels", ["lab", "rot_lab"])):
                for name in names:
                    np.testing.assert_array_equal(
                        batch[kind][name][row],
                        _rasterized(transformed[name], x0, y0, size, k),
                        err_msg=f"{name}, cell {cell}, k={k}",
                    )
            compared += 1
    assert compared == 120
    assert covered >= 30, covered  # the comparison is not between empty tiles

    meta = next(iter(collection)).attrs["spatialbatch"]
    assert set(meta["resampled_from_level"]) == (
        {"rot", "rot_lab", "frac"} | ({"img", "lab", "shift"} if k != 1 else set())
    )
    if k == 0.25:
        assert meta["resampled_from_level"]["rot"] == 1


def test_whole_pixel_shifts_are_copied_with_their_origin(transformed, tmp_path):
    collection = _write(transformed, tmp_path, 1.0)
    group = next(iter(collection))
    meta = group.attrs["spatialbatch"]
    assert meta["origins"]["shift"] == [5, -7] and meta["origins"]["img"] == [0, 0]
    np.testing.assert_array_equal(group["images"]["shift"][...], group["images"]["img"][...])  # bytes as stored
    rot = group["images"]["rot"]
    assert rot.shards == (3, 64, 64) and rot.dtype == np.uint8
    x0, y0 = meta["origins"]["rot"]
    assert x0 >= -TILE_UNITS and y0 >= -TILE_UNITS  # only the region that tiles reach is stored


def test_collections_without_origins_still_load(synthetic, tmp_path, ref_tile):
    """Datasets written before origins were recorded start at pixel (0, 0)."""
    sdata, raw = synthetic
    collection = SpatialDatasetCollection(tmp_path / "old.zarr")
    collection.add_sdatas(
        [sdata.path], images=["img"], tile_dim_in_units=TILE_UNITS, image_chunk=32, image_shard=64, **KW
    )
    group = next(iter(collection))
    meta = dict(group.attrs["spatialbatch"])
    del meta["origins"]
    group.update_attributes({"spatialbatch": meta})

    loader = Loader(images=["img"], batch_size=50, chunk_size=50, preload_nchunks=1).use_collection(
        SpatialDatasetCollection(zarr.open_group(tmp_path / "old.zarr", mode="r"), mode="r")
    )
    batch = next(iter(loader))
    cell = int(str(batch["obs"]["instance_id"].iloc[0]).split("-")[1])
    np.testing.assert_array_equal(batch["images"]["img"][0], ref_tile(raw["image"], *raw["centers"][cell]))


def test_resampling_errors(synthetic, tmp_path):
    from spatialbatch import write_sharded_sdata

    sdata, _ = synthetic
    group = zarr.open_group(tmp_path / "x.zarr", mode="w")
    with pytest.raises(ValueError, match="integer number of pixels"):
        write_sharded_sdata(
            group, sdata, key="dataset_0", images=["img"], tile_dim_in_units=32, target_unit_to_pixels=0.3, **KW
        )
    with pytest.raises(ValueError, match="positive"):
        write_sharded_sdata(
            group, sdata, key="dataset_0", images=["img"], tile_dim_in_units=32, target_unit_to_pixels=0, **KW
        )
