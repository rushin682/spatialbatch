"""How many pixels a request reads per spot for different row orders and chunk sizes, with no image IO.

``write_sharded_sdata`` used to sort the rows along a Morton curve of the tile *centers*, but the loader groups tiles into
windows by the cell of their *start* (``Loader._fetch_tiles``), which is the center minus half a tile. This replays the
sampler (random chunks of consecutive rows) on the stored tile origins of a collection under either order, and counts the
windows and decoded bytes with the loader's own grouping (:func:`profile_request.window_cost`).

Usage: ``simulate_order.py <split> [--chunk-sizes 16 32 64] [--blocks 512 1024 2048]``
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import zarr
from profile_request import window_cost
from slides import HE_ALIGNED, OUT_ROOT, SUFFIX, collection_path

from spatialbatch import SpatialDatasetCollection
from spatialbatch.utils import morton_order


def main() -> None:
    """Print windows per request and KB read per spot for each order and chunk size."""
    parser = argparse.ArgumentParser()
    parser.add_argument("split", choices=["train", "val", "test"])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--blocks", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--spots-per-request", type=int, default=4096)
    parser.add_argument("--requests", type=int, default=10, help="Requests drawn per dataset.")
    args = parser.parse_args()

    collection = SpatialDatasetCollection(zarr.open_group(collection_path(args.split), mode="r"))
    rng = np.random.default_rng(0)
    rows = []
    for key in ("center", "start"):
        for block in args.blocks:
            for chunk_size in args.chunk_sizes:
                n_bytes = n_windows = n_spots = n_requests = 0
                for group in collection:
                    origin = np.asarray(group["obsm"]["tile_origin"][:]).astype(np.int64)
                    height, width = group.attrs["spatialbatch"]["tile_hw"]
                    array = group["images"][HE_ALIGNED]
                    n_chunks = len(origin) // chunk_size
                    per_request = args.spots_per_request // chunk_size
                    if n_chunks < per_request:
                        continue
                    # the orders are recomputed from the stored origins: the result does not depend on the order on disk
                    x0, y0 = origin[:, 0], origin[:, 1]
                    centers = (x0 + width // 2, y0 + height // 2)
                    order = morton_order(*(centers if key == "center" else (x0, y0)), block)
                    x0, y0 = x0[order], y0[order]
                    for _ in range(args.requests):
                        starts = rng.choice(n_chunks, per_request, replace=False) * chunk_size
                        picked = (starts[:, None] + np.arange(chunk_size)).ravel()
                        windows, cost = window_cost(x0[picked], y0[picked], (height, width), array.shape[-2:], 3)
                        n_bytes, n_windows, n_spots, n_requests = (
                            n_bytes + cost,
                            n_windows + len(windows),
                            n_spots + len(picked),
                            n_requests + 1,
                        )
                rows.append(
                    {
                        "order_by": key,
                        "block_px": block,
                        "chunk_size": chunk_size,
                        "windows_per_request": n_windows / n_requests,
                        "read_kb_per_spot": n_bytes / n_spots / 1e3,
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
    (OUT_ROOT / "results" / f"simulate_order_{args.split}{SUFFIX}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
