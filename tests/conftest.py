from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# The real Xenium stores used by the parity tests and the benchmarks.
DATA_DIR = Path(os.environ.get("SPATIALBATCH_DATA", Path(__file__).parents[2] / "sddb-downloads"))
STORES = ["6T1aLES2sndpL1bQ0001.zarr", "1LYWTr2DRrkM6NTY0001.zarr", "ftcwfWqbDwNk5EMp0001.zarr"]


def pytest_collection_modifyitems(items):
    for item in items:
        if "slow" in item.keywords and not (DATA_DIR / STORES[0]).exists():
            item.add_marker(pytest.mark.skip(reason=f"real stores not found in {DATA_DIR} (set SPATIALBATCH_DATA)"))


@pytest.fixture(scope="session", params=STORES)
def store_path(request) -> Path:
    return DATA_DIR / request.param


@pytest.fixture(scope="session")
def sdata(store_path):
    import spatialdata as sd

    return sd.read_zarr(store_path)


# -- a small synthetic, Xenium-shaped SpatialData for fast tests ---------------------------------------------------

TILE = 32
SCALE = 2.0  # circles -> global, like Xenium's micron -> pixel


def reference_tile(image: np.ndarray, cx: float, cy: float, size: int = TILE) -> np.ndarray:
    """Independent reference: the pixels under a tile centered on (cx, cy), zero-padded outside the array."""
    x0, y0 = int(np.floor(cx - size / 2 + 0.5)), int(np.floor(cy - size / 2 + 0.5))
    height, width = image.shape[-2:]
    out = np.zeros((*image.shape[:-2], size, size), dtype=image.dtype)
    cy0, cy1, cx0, cx1 = max(y0, 0), min(y0 + size, height), max(x0, 0), min(x0 + size, width)
    out[..., cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = image[..., cy0:cy1, cx0:cx1]
    return out


def make_sdata(
    path: Path, *, n_cells: int = 200, seed: int = 0, shape: tuple[int, int] = (300, 340), radius=3.0, channels: int = 1
):
    """Write (and read back) a SpatialData with the layout of a Xenium store. Returns it with its raw arrays."""
    import anndata as ad
    import pandas as pd
    import spatialdata as sd
    from scipy import sparse as sp
    from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel, TableModel
    from spatialdata.transformations import Identity, Scale

    rng = np.random.default_rng(seed)
    height, width = shape
    image = rng.integers(1, 60000, (channels, height, width), dtype=np.uint16)
    labels = rng.integers(0, 50, (height, width)).astype(np.uint32)
    centers = rng.uniform([0, 0], [width, height], (n_cells, 2))  # global pixels; some tiles cross the border
    ids = np.array([f"cell{seed}-{i}" for i in range(n_cells)])
    radii = np.full(n_cells, radius) if np.isscalar(radius) else radius
    circles = ShapesModel.parse(
        centers / SCALE,
        geometry=0,
        radius=radii,
        index=ids,
        transformations={"global": Scale([SCALE, SCALE], axes=("x", "y"))},
    )
    adata = ad.AnnData(
        sp.random(n_cells, 20, density=0.3, format="csr", dtype=np.float32, random_state=seed),
        obs=pd.DataFrame(
            {"instance_id": ids, "region": pd.Categorical(["cells"] * n_cells), "label": np.arange(n_cells) + 1},
            index=ids,
        ),
    )
    adata.obs_names = ids
    adata.obsm["spatial"] = (centers / SCALE).astype(np.float32)
    sdata = sd.SpatialData(
        images={
            "img": Image2DModel.parse(
                image, dims=("c", "y", "x"), scale_factors=[2], transformations={"global": Identity()}
            )
        },
        labels={
            "lab": Labels2DModel.parse(
                labels, dims=("y", "x"), scale_factors=[2], transformations={"global": Identity()}
            )
        },
        shapes={"cells": circles},
        tables={"table": TableModel.parse(adata, region="cells", region_key="region", instance_key="instance_id")},
    )
    sdata.write(path)
    return sd.read_zarr(path), {"image": image, "labels": labels, "centers": centers, "ids": ids}


@pytest.fixture
def synthetic(tmp_path):
    return make_sdata(tmp_path / "a.zarr")


@pytest.fixture
def make():
    """Factory for synthetic SpatialData stores: ``make(path, n_cells=..., seed=..., ...) -> (sdata, raw)``."""
    return make_sdata


@pytest.fixture
def ref_tile():
    return reference_tile
