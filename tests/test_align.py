from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import spatialdata as sd
import zarr
from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel
from spatialdata.transformations import Affine, Identity, Scale, Sequence, Translation, get_transformation
from xarray import DataArray, DataTree

from spatialbatch import Loader, align_to_cs, write_sharded_sdata
from spatialbatch.utils import element_to_cs_matrix, is_identity_xy

TILE = 32
XY = ("x", "y")
# a registration-like transformation: rotation, anisotropic scaling, shear and a fractional translation
GENERAL = np.array([[0.93, -0.41, 61.3], [0.37, 1.08, -17.6], [0.0, 0.0, 1.0]])
_C, _S = np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25))
ROTATE_25 = Affine(np.array([[_C, -_S, 0.0], [_S, _C, 0.0], [0.0, 0.0, 1.0]]), input_axes=XY, output_axes=XY)


def _add(sdata, name, data, transformation, *, model=Image2DModel, scale_factors=None, extra=None):
    """Add a raster element with `transformation` to ``global`` to the store of `sdata`, and read the store back."""
    dims = ("c", "y", "x") if data.ndim == 3 else ("y", "x")
    sdata[name] = model.parse(
        data, dims=dims, scale_factors=scale_factors, transformations={"global": transformation, **(extra or {})}
    )
    sdata.write_element(name)
    return sd.read_zarr(sdata.path)


def _scale0(element) -> np.ndarray:
    level = cast("DataArray", sd.get_pyramid_levels(element, n=0) if isinstance(element, DataTree) else element)
    return np.asarray(level.values)


@pytest.fixture
def he(synthetic):
    """The synthetic store with an RGB ``he`` image that is not in the pixel grid of ``global``."""
    sdata, raw = synthetic
    rng = np.random.default_rng(1)
    image = rng.integers(1, 255, (3, 200, 260), dtype=np.uint8)
    transformation = Affine(GENERAL, input_axes=XY, output_axes=XY)
    return _add(sdata, "he", image, transformation, scale_factors=[2]), raw, image


def test_integer_translation_is_exact(synthetic):
    sdata, raw = synthetic
    sdata = _add(sdata, "moved", raw["image"], Translation([3, 2], axes=XY))
    out = _scale0(align_to_cs(sdata.path, "moved")["moved_aligned"])
    channels, height, width = raw["image"].shape
    expected = np.zeros((channels, height + 2, width + 3), raw["image"].dtype)
    expected[:, 2:, 3:] = raw["image"]
    np.testing.assert_array_equal(out, expected)


def test_quarter_turn_is_exact(synthetic):
    sdata, raw = synthetic
    labels = raw["labels"]
    height, width = labels.shape
    # (x, y) -> (height - y, x): a quarter turn that lands at positive coordinates
    quarter = Affine(np.array([[0, -1, height], [1, 0, 0], [0, 0, 1]]), input_axes=XY, output_axes=XY)
    sdata = _add(sdata, "turned", labels, quarter, model=Labels2DModel)
    out = _scale0(align_to_cs(sdata.path, "turned")["turned_aligned"])
    assert out.shape == (width, height)
    # spatialdata puts pixel centers at integer coordinates: column j of the output samples row `height - j`
    np.testing.assert_array_equal(out[:, 1:], np.rot90(labels, k=-1)[:, :-1])
    assert (out[:, 0] == 0).all()


def test_labels_keep_their_ids_and_dtype(synthetic):
    sdata, raw = synthetic
    sdata = _add(sdata, "big", raw["labels"], Scale([2.5, 1.7], axes=XY), model=Labels2DModel)
    out = _scale0(align_to_cs(sdata.path, "big")["big_aligned"])
    assert out.dtype == raw["labels"].dtype
    assert out.shape == (int(np.ceil(raw["labels"].shape[0] * 1.7)), int(np.ceil(raw["labels"].shape[1] * 2.5)))
    assert set(np.unique(out)) <= set(np.unique(raw["labels"])) | {0}


@pytest.mark.parametrize("model", [Image2DModel, Labels2DModel], ids=["image", "labels"])
def test_blocks_equal_one_rasterization_of_the_whole_extent(synthetic, model):
    """Resampling block by block (small blocks, so many seams) gives the pixels of a single full rasterization."""
    sdata, raw = synthetic
    data = raw["image"] if model is Image2DModel else raw["labels"]
    sdata = _add(sdata, "he", data, Affine(GENERAL, input_axes=XY, output_axes=XY), model=model)
    out = _scale0(align_to_cs(sdata.path, "he", chunk=37)["he_aligned"])
    full = sd.rasterize(
        sdata["he"],
        axes=XY,
        min_coordinate=[0, 0],
        max_coordinate=[out.shape[-1], out.shape[-2]],
        target_coordinate_system="global",
        target_unit_to_pixels=1.0,
    )
    expected = np.asarray(full.values)
    np.testing.assert_array_equal(out, expected if model is Image2DModel else expected[0])
    assert out.dtype == data.dtype


def test_written_chunks_are_those_of_the_original(synthetic):
    sdata, raw = synthetic
    sdata["he"] = Image2DModel.parse(
        raw["image"],
        dims=("c", "y", "x"),
        chunks={"c": 1, "y": 64, "x": 128},
        transformations={"global": Affine(GENERAL, input_axes=XY, output_axes=XY)},
    )
    sdata.write_element("he")
    align_to_cs(sdata.path, "he", chunk=100)
    assert zarr.open_array(sdata.path / "images" / "he_aligned" / "s0", mode="r").chunks == (1, 64, 128)


def test_aligned_element_takes_the_fast_path(he, tmp_path):
    sdata, _, _ = he
    with pytest.raises(NotImplementedError, match="align_to_cs"):
        write_sharded_sdata(
            zarr.open_group(tmp_path / "c0.zarr", mode="w"), sdata, key="d", region="cells", images=["he"]
        )

    sdata = align_to_cs(sdata.path, "he")
    element = sdata["he_aligned"]
    assert is_identity_xy(element_to_cs_matrix(element, "global"))
    assert sd.models.get_model(element) is Image2DModel
    loader = Loader(images=["he_aligned"], tile_dim_in_units=TILE).add_sdata(sdata, region="cells")
    assert loader._items[0].sources["he_aligned"].fallback is None
    write_sharded_sdata(
        zarr.open_group(tmp_path / "c1.zarr", mode="w"),
        sdata,
        key="d",
        region="cells",
        images=["he_aligned"],
        tile_dim_in_units=TILE,
        image_chunk=32,
        image_shard=128,
    )


@pytest.mark.parametrize(
    "transformation",
    [
        pytest.param(Affine(GENERAL, input_axes=XY, output_axes=XY), id="affine"),
        pytest.param(
            Sequence([Scale([1.3, 0.8], axes=XY), ROTATE_25, Translation([40.5, 12.25], axes=XY)]),
            id="sequence",
        ),
    ],
)
def test_fast_path_tiles_are_rasterized_tiles_of_the_original(synthetic, transformation):
    """A fast-path tile of the aligned element is :func:`spatialdata.rasterize` of the original over that tile.

    The tile is the one the fast path reads, starting at the integer pixel ``round_half_up(min corner)`` as for every
    element already in the grid. ``Loader(rasterize=True)`` instead resamples the original from the fractional min
    corner, so its tiles are offset by up to half a pixel and differ from these by a sub-pixel shift.
    """
    sdata, raw = synthetic
    image = np.random.default_rng(2).integers(1, 255, (3, 180, 240), dtype=np.uint8)
    sdata = align_to_cs(_add(sdata, "he", image, transformation).path, "he")
    fast = Loader(
        images=["he_aligned"],
        tile_dim_in_units=TILE,
        return_index=True,
        batch_size=50,
        chunk_size=50,
        preload_nchunks=1,
    ).add_sdata(sdata, region="cells")
    seen = 0
    for batch in fast:
        for row, idx in enumerate(batch["index"]):
            cx, cy = raw["centers"][idx]
            x0, y0 = np.floor(cx - TILE / 2 + 0.5), np.floor(cy - TILE / 2 + 0.5)
            reference = sd.rasterize(
                sdata["he"],
                axes=XY,
                min_coordinate=[x0, y0],
                max_coordinate=[x0 + TILE, y0 + TILE],
                target_coordinate_system="global",
                target_unit_to_pixels=1.0,
            )
            reference = np.asarray(reference.values)
            # the aligned array starts at the origin, so the rasterized tile is zero where it reaches below it
            reference[:, : max(0, -int(y0)), :] = 0
            reference[:, :, : max(0, -int(x0))] = 0
            np.testing.assert_array_equal(batch["images"]["he_aligned"][row], reference)
            seen += 1
    assert seen == 200


def test_source_element_is_unchanged_and_overwrite(he):
    sdata, _, image = he
    before = get_transformation(sdata["he"], get_all=True)
    align_to_cs(sdata.path, "he")
    with pytest.raises(ValueError, match="overwrite"):
        align_to_cs(sdata.path, "he")
    out = align_to_cs(sdata.path, "he", overwrite=True)
    np.testing.assert_array_equal(_scale0(out["he"]), image)
    assert get_transformation(out["he"], get_all=True).keys() == before.keys()
    assert "he_aligned" in out and "img" in out


def test_new_store_replaces_in_place(he, tmp_path):
    sdata, _, image = he
    out = align_to_cs(sdata.path, ["he"], out_path=tmp_path / "out.zarr")
    assert is_identity_xy(element_to_cs_matrix(out["he"], "global"))
    assert "he_aligned" not in out
    assert set(sd.read_zarr(sdata.path).images) == set(out.images)
    assert set(out.labels) == {"lab"} and set(out.shapes) == {"cells"} and set(out.tables) == {"table"}
    # the source store is not modified
    source = sd.read_zarr(sdata.path)
    np.testing.assert_array_equal(_scale0(source["he"]), image)
    assert not is_identity_xy(element_to_cs_matrix(source["he"], "global"))
    with pytest.raises(ValueError, match="exists"):
        align_to_cs(sdata.path, "he", out_path=tmp_path / "out.zarr")
    align_to_cs(sdata.path, "he", out_path=tmp_path / "out.zarr", overwrite=True)
    with pytest.raises(ValueError, match="differ"):
        align_to_cs(sdata.path, "he", out_path=sdata.path)


def test_like_gives_the_reference_extent(he):
    sdata, raw, _ = he
    out = align_to_cs(sdata.path, "he", like="img")
    assert _scale0(out["he_aligned"]).shape == (3, *raw["image"].shape[-2:])


def test_like_must_be_aligned(he):
    sdata, _, _ = he
    with pytest.raises(ValueError, match="like"):
        align_to_cs(sdata.path, "he", like="he")


def test_identity_element_is_skipped(synthetic):
    sdata, _ = synthetic
    with pytest.warns(UserWarning, match="already in the pixel grid"):
        out = align_to_cs(sdata.path, "img")
    assert "img_aligned" not in out


def test_invalid_elements(synthetic):
    sdata, _ = synthetic
    volume = np.zeros((1, 4, 10, 10), np.uint8)
    sdata["vol"] = Image3DModel.parse(volume, dims=("c", "z", "y", "x"), transformations={"global": Identity()})
    sdata.write_element("vol")
    with pytest.raises(NotImplementedError, match="3D"):
        align_to_cs(sdata.path, "vol")
    with pytest.raises(ValueError, match="image or labels"):
        align_to_cs(sdata.path, "cells")
    sdata = _add(sdata, "flat", np.ones((1, 10, 10), np.uint8), Scale([1.0, 0.0], axes=XY))
    with pytest.raises(ValueError, match="invertible"):
        align_to_cs(sdata.path, "flat")
    sdata = _add(sdata, "moved", np.ones((1, 10, 10), np.uint8), Translation([2, 2], axes=XY))
    with pytest.raises(ValueError, match="coordinate system"):
        align_to_cs(sdata.path, "moved", cs="nowhere")


def test_pyramid_is_kept_by_default(synthetic):
    sdata, raw = synthetic
    sdata = _add(sdata, "deep", raw["image"], Translation([1, 1], axes=XY), scale_factors=[2, 2])
    sdata = _add(sdata, "flat", raw["image"], Translation([1, 1], axes=XY))
    out = align_to_cs(sdata.path, ["deep", "flat"])
    assert len(sd.get_pyramid_levels(out["deep_aligned"])) == 3
    assert isinstance(out["flat_aligned"], DataArray)
    out = align_to_cs(sdata.path, "deep", scale_factors=[4], overwrite=True)
    shapes = sd.get_pyramid_levels(out["deep_aligned"], attr="shape")
    assert len(shapes) == 2 and shapes[1][-1] == shapes[0][-1] // 4


def test_other_coordinate_systems_keep_their_placement(synthetic):
    sdata, raw = synthetic
    to_global = Affine(GENERAL, input_axes=XY, output_axes=XY)
    to_other = Sequence([Scale([0.5, 0.25], axes=XY), Translation([7, -3], axes=XY)])
    sdata = _add(sdata, "he", raw["image"], to_global, extra={"microns": to_other})
    out = align_to_cs(sdata.path, "he")["he_aligned"]
    # a pixel of the aligned element is a point of `global`: it lands in `microns` where that point of the original did
    original = to_other.to_affine_matrix(XY, XY) @ np.linalg.inv(to_global.to_affine_matrix(XY, XY))
    np.testing.assert_allclose(element_to_cs_matrix(out, "microns"), original, atol=1e-9)
