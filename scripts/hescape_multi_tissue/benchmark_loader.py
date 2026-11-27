"""Throughput of the multi-modal loader on a collection, as ranks of one node (``torchrun --nproc-per-node N``).

Each rank streams its shard of the collection with a per-rank batch of ``global_batch // world_size``, moves the
512 px H&E tiles to its GPU, resizes them to 224 there and densifies the expression, then "trains" for ``--step-ms``
milliseconds. It reports how many spots per second the node consumes and which fraction of the time the GPU waited for
data: with ``--step-ms`` close to the real step time that is the stall the loader would cost.

Usage: ``torchrun --nproc-per-node 4 benchmark_loader.py <split> --mode bare|workers [--chunk-size 32 ...]``
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import zarr
from annbatch.samplers import DistributedSampler
from slides import HE_ALIGNED, OUT_ROOT, SUFFIX, collection_path
from torch.utils.data import DataLoader

from spatialbatch import Loader, SpatialDatasetCollection
from spatialbatch.samplers import RandomSampler

warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part in the Zarr format 3.*")

parser = argparse.ArgumentParser()
parser.add_argument("split", choices=["train", "val", "test"])
parser.add_argument("--mode", choices=["bare", "workers"], default="bare")
parser.add_argument("--workers", type=int, default=2, help="DataLoader workers per rank (mode workers).")
parser.add_argument("--global-batch", type=int, default=1024)
parser.add_argument("--chunk-size", type=int, default=32)
parser.add_argument("--preload-nchunks", type=int, default=128)
parser.add_argument("--input-px", type=int, default=224, help="Side the 512 px tiles are resized to on the GPU.")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--warmup", type=int, default=4)
parser.add_argument("--step-ms", type=float, default=0.0, help="Simulated compute of a training step.")
parser.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--tag", default="")


def main() -> None:
    """Run on every rank. The guard below keeps DataLoader workers (forkserver on Python 3.14) from re-running it."""
    args = parser.parse_args()

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)

    cpus = len(os.sched_getaffinity(0)) // world
    zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "threading.max_workers": max(cpus // (args.workers if args.mode == "workers" else 1), 1),
        }
    )

    collection = SpatialDatasetCollection(zarr.open_group(collection_path(args.split), mode="r"))
    sampler = DistributedSampler(
        RandomSampler(
            chunk_size=args.chunk_size,
            preload_nchunks=args.preload_nchunks,
            batch_size=args.global_batch // world,
            drop_last=True,
            rng=np.random.default_rng(0),
        ),
        dist_info="torch" if world > 1 else (lambda: (0, 1)),
    )
    loader = Loader(
        images=[HE_ALIGNED], return_annotations=["sample_id"], batch_sampler=sampler, prefetch=args.prefetch, to="torch"
    ).use_collection(collection)
    source = (
        loader
        if args.mode == "bare"
        else DataLoader(
            loader,
            batch_size=None,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )
    )

    wait, total_spots, steps = 0.0, 0, 0
    start = time.perf_counter()
    ready = time.perf_counter()
    for batch in source:
        now = time.perf_counter()
        if steps >= args.warmup:
            wait += now - ready
        tiles = batch["images"][HE_ALIGNED].to(device, non_blocking=True)
        x = batch["X"].to(device, non_blocking=True).to_dense()
        tiles = torch.nn.functional.interpolate(
            tiles.float(), size=args.input_px, mode="bilinear", antialias=True
        )  # (batch, 3, input_px, input_px)
        x = torch.log1p(x / x.sum(dim=1, keepdim=True).clamp(min=1) * 1e4)  # (batch, n_genes)
        torch.cuda.synchronize()
        time.sleep(args.step_ms / 1000)
        steps += 1
        if steps == args.warmup:
            start = time.perf_counter()
        elif steps > args.warmup:
            total_spots += tiles.shape[0]
        if steps == args.warmup + args.steps:
            break
        ready = time.perf_counter()
    elapsed = time.perf_counter() - start

    stats = torch.tensor(
        [total_spots, elapsed, wait, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6], device=device
    )
    gathered = [torch.zeros_like(stats) for _ in range(world)]
    dist.all_gather(gathered, stats) if world > 1 else gathered.__setitem__(0, stats)
    if rank == 0:
        per_rank = torch.stack(gathered).cpu().numpy()
        result = {
            "tag": args.tag,
            "split": args.split,
            "collection": f"{args.split}{SUFFIX}",
            "mode": args.mode,
            "prefetch": args.prefetch,
            "workers": args.workers if args.mode == "workers" else 0,
            "world_size": world,
            "global_batch": args.global_batch,
            "chunk_size": args.chunk_size,
            "preload_nchunks": args.preload_nchunks,
            "input_px": args.input_px,
            "step_ms": args.step_ms,
            "steps": args.steps,
            "spots_per_s_node": float(per_rank[:, 0].sum() / per_rank[:, 1].max()),
            "stall_fraction": float((per_rank[:, 2] / per_rank[:, 1]).mean()),
            "rss_gb_max_rank": float(per_rank[:, 3].max()),
        }
        print(json.dumps(result), flush=True)
        with (OUT_ROOT / "results" / "benchmark.jsonl").open("a") as f:
            f.write(json.dumps(result) + "\n")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
