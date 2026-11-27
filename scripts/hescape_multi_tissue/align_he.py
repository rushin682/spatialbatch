"""Resample the H&E image of one slide onto the pixel grid of ``global``, as ``he_image_aligned`` in the same store.

Usage: ``align_he.py <index>`` where index selects one slide of ``slides.slides()`` (train, val, test in this order),
so that a SLURM job array of ``0..N-1`` aligns every slide. ``--list`` prints the slides and their indices.
"""

from __future__ import annotations

import argparse
import time
import warnings

import dask
from slides import HE_ALIGNED, HE_IMAGE, LIKE, slides

from spatialbatch import align_to_cs

warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part in the Zarr format 3.*")

parser = argparse.ArgumentParser()
parser.add_argument("index", type=int, nargs="?")
parser.add_argument("--list", action="store_true")
parser.add_argument("--workers", type=int, default=32, help="Threads of the dask scheduler.")
args = parser.parse_args()

stores = [path for split in slides().values() for path in split]
if args.list:
    print(*(f"{i:2d} {path.parent.name}/{path.stem}" for i, path in enumerate(stores)), sep="\n")
    raise SystemExit
path = stores[args.index]

if (path / "images" / HE_ALIGNED).exists():
    print(f"{path.stem}: {HE_ALIGNED} exists already, skipped")
    raise SystemExit
start = time.perf_counter()
with dask.config.set(scheduler="threads", num_workers=args.workers):
    align_to_cs(path, HE_IMAGE, like=LIKE, chunk=4096)
print(f"{path.parent.name}/{path.stem}: aligned in {time.perf_counter() - start:.0f}s", flush=True)
