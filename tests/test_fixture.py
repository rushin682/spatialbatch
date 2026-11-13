"""The published sddb loader-contract fixture: download, check its sha256 and read one batch.

Marked ``network``: deselected by default and in CI (``addopts`` in pyproject.toml). Run with
``pytest -m network tests/test_fixture.py``.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile

import numpy as np
import pytest

from spatialbatch import Loader, SpatialDatasetCollection

pytestmark = pytest.mark.network

URL = "https://exampledata.scverse.org/spatialdata-db/fixtures/sddb-contract/0.1.0/atera-breast-crop.zarr.zip"
SHA256 = "7d0ea895d7a4a663398486a564d53ff6ce7056b1026c53b8a49516b3dfaadc5e"
ROOT = "sddb-contract-0.1.0-atera-breast-crop.zarr"  # the one top-level folder of the zip


def test_fixture_one_batch(tmp_path):
    archive = tmp_path / "fixture.zip"
    # the host refuses urllib's default User-Agent with a 403
    request = urllib.request.Request(URL, headers={"User-Agent": "spatialbatch-tests"})
    with urllib.request.urlopen(request, timeout=60) as response, archive.open("wb") as fh:
        shutil.copyfileobj(response, fh)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SHA256
    with zipfile.ZipFile(archive) as z:
        z.extractall(tmp_path)

    loader = Loader(images=["he"], labels=["cells"], crop_px=224, batch_size=32, shuffle=True)
    batch = next(iter(loader.use_collection(SpatialDatasetCollection(tmp_path / ROOT))))
    he, cells = np.asarray(batch["images"]["he"]), np.asarray(batch["labels"]["cells"])
    assert he.shape == (32, 3, 224, 224) and he.dtype == np.uint8
    assert cells.shape == (32, 224, 224)
    assert batch["X"].shape[0] == 32
