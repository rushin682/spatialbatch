"""Where the time of one loader request goes, and how many pixels it reads per spot (CPU only).

A request is ``preload_nchunks`` chunks of ``chunk_size`` consecutive spots. For each configuration this times the
expression and the pixel reads of the same request separately, and counts what ``Loader._fetch_tiles`` reads from the
image: the windows (one per ``_WINDOW_PX`` cell of tile starts) and their decoded bytes per spot.

Usage: ``profile_request.py <split> [--threads 32 64 128] [--chunk-sizes 16 32 64]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import warnings

import numpy as np
import zarr
import zarr.core.sync as zsync
from slides import HE_ALIGNED, OUT_ROOT, SUFFIX, collection_path

from spatialbatch import Loader, SpatialDatasetCollection
from spatialbatch.loader import _WINDOW_PX

warnings.filterwarnings("ignore")


def window_cost(
    x0: np.ndarray, y0: np.ndarray, tile_hw: tuple[int, int], array_hw: tuple[int, int], channels: int
) -> tuple[list[tuple[int, int, int, int]], int]:
    """The windows ``Loader._fetch_tiles`` reads for tiles starting at ``(x0, y0)``, and their decoded bytes.

    Tiles are grouped by the ``_WINDOW_PX`` cell of their (clipped) start, and one window, the bounding box of the group,
    is read per group. Returns the windows as ``(y_start, y_stop, x_start, x_stop)``.
    """
    (th, tw), (height, width) = tile_hw, array_hw
    cy0, cy1 = np.clip(y0, 0, height), np.clip(y0 + th, 0, height)
    cx0, cx1 = np.clip(x0, 0, width), np.clip(x0 + tw, 0, width)
    cells = (cy0 // _WINDOW_PX) * (width // _WINDOW_PX + 2) + cx0 // _WINDOW_PX
    windows = []
    for cell in np.unique(cells):
        group = cells == cell
        windows.append((int(cy0[group].min()), int(cy1[group].max()), int(cx0[group].min()), int(cx1[group].max())))
    return windows, sum((y1 - y0_) * (x1 - x0_) * channels for y0_, y1, x0_, x1 in windows)


def request_windows(
    loader: Loader, dataset_index_to_rows: dict[int, np.ndarray]
) -> list[tuple[int, int, int, int, int]]:
    """The windows of one request, as ``(dataset, y_start, y_stop, x_start, x_stop)``."""
    out = []
    for index, rows in dataset_index_to_rows.items():
        item = loader._items[index]
        geometry, array = item.geometry, item.sources[HE_ALIGNED].array
        windows, _ = window_cost(
            geometry.x0[rows], geometry.y0[rows], (geometry.height, geometry.width), array.shape[-2:], array.shape[0]
        )
        out += [(index, *window) for window in windows]
    return out


async def read_windows(loader: Loader, windows: list[tuple[int, int, int, int, int]]) -> None:
    """Only the decoding: read the windows concurrently, with no cropping of tiles out of them."""
    await asyncio.gather(
        *(
            loader._items[index]
            .sources[HE_ALIGNED]
            .array._async_array.getitem((slice(None), slice(y0, y1), slice(x0, x1)))
            for index, y0, y1, x0, x1 in windows
        )
    )


def main() -> None:
    """Profile each configuration and append the results to ``results/profile_request.jsonl``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "val", "test"])
    parser.add_argument("--threads", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--spots-per-request", type=int, default=4096)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
    collection = SpatialDatasetCollection(zarr.open_group(collection_path(args.split), mode="r"))
    for chunk_size in args.chunk_sizes:
        loader = Loader(
            images=[HE_ALIGNED],
            batch_size=256,
            chunk_size=chunk_size,
            preload_nchunks=args.spots_per_request // chunk_size,
            shuffle=True,
            rng=np.random.default_rng(0),
        ).use_collection(collection)
        requests = []
        for request in loader._batch_sampler.sample(loader.n_obs):
            requests.append(loader._requests_to_dataset_rows(request["requests"])[0])
            if len(requests) == args.repeats:
                break
        n_spots = sum(len(rows) for rows in requests[0].values())
        windows = request_windows(loader, requests[0])
        n_bytes = sum((y1 - y0) * (x1 - x0) * 3 for _, y0, y1, x0, x1 in windows)
        for threads in args.threads:
            zarr.config.set({"threading.max_workers": threads})
            seconds = {"pixels": [], "expression": [], "split": [], "decode": [], "decode_warm": []}
            for rows in requests:
                start = time.perf_counter()
                zsync.sync(read_windows(loader, request_windows(loader, rows)))
                seconds["decode"].append(time.perf_counter() - start)
                # the same windows again: now in the page cache, so what is left is decoding
                start = time.perf_counter()
                zsync.sync(read_windows(loader, request_windows(loader, rows)))
                seconds["decode_warm"].append(time.perf_counter() - start)
                start = time.perf_counter()
                buffers = zsync.sync(loader._index_pixels(rows))
                seconds["pixels"].append(time.perf_counter() - start)
                # what `_iter` does next: the in-memory shuffle, one fancy-indexing copy per batch (single thread)
                order = np.random.default_rng(0).permutation(n_spots)
                start = time.perf_counter()
                for first in range(0, n_spots, 256):
                    buffers[HE_ALIGNED][order[first : first + 256]]
                seconds["split"].append(time.perf_counter() - start)
                del buffers
                start = time.perf_counter()
                zsync.sync(loader._index_datasets(rows))
                seconds["expression"].append(time.perf_counter() - start)
            # the first request opens files and fills caches: report the median of all
            pixels, expression, split, decode, decode_warm = (
                float(np.median(seconds[k])) for k in ("pixels", "expression", "split", "decode", "decode_warm")
            )
            result = {
                "tag": args.tag,
                "split": args.split,
                "collection": f"{args.split}{SUFFIX}",
                "chunk_size": chunk_size,
                "spots": n_spots,
                "threads": threads,
                "windows_per_request": len(windows),
                "read_kb_per_spot": n_bytes / n_spots / 1e3,
                "pixels_s": pixels,
                "expression_s": expression,
                "split_s": split,
                "decode_s": decode,
                "decode_warm_s": decode_warm,
                "decoded_mb_per_s": n_bytes / pixels / 1e6,
                "spots_per_s_pixels": n_spots / pixels,
            }
            print(json.dumps(result), flush=True)
            with (OUT_ROOT / "results" / "profile_request.jsonl").open("a") as f:
                f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
