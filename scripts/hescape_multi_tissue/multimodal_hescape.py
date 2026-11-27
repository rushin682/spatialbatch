"""The multimodal tutorial on the HESCAPE collections: H&E tiles with the expression of their 55 um spot.

Streams (``he_image_aligned``, ``spots_55um_table``) pairs from ``train.zarr`` with a global batch of 1024 spread over
the ranks of ``torchrun``, trains the tutorial's small CNN to predict the log-normalised expression from the tile (DDP)
and evaluates on ``val.zarr``. Writes the figures and the losses to ``results/``.

Usage: ``torchrun --nproc-per-node 4 multimodal_hescape.py --input crop|resize``
"""

from __future__ import annotations

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spatialdata as sd
import torch
import torch.distributed as dist
import zarr
from annbatch.samplers import DistributedSampler
from slides import HE_ALIGNED, OUT_ROOT, REGION, SUFFIX, TILE_PX, collection_path, slides
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from spatialbatch import Loader, SpatialDatasetCollection
from spatialbatch.samplers import RandomSampler, SequentialSampler

warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part in the Zarr format 3.*")

parser = argparse.ArgumentParser()
parser.add_argument("--input", choices=["crop", "resize"], default="resize", help="512 px tile -> 224 px model input.")
parser.add_argument("--global-batch", type=int, default=1024)
parser.add_argument("--chunk-size", type=int, default=32)
parser.add_argument("--preload-nchunks", type=int, default=128)
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--val-steps", type=int, default=20)
parser.add_argument("--workers", type=int, default=2)


def main() -> None:
    """Run on every rank. The guard below keeps DataLoader workers (forkserver on Python 3.14) from re-running it."""
    args = parser.parse_args()

    MODEL_PX = 224
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    zarr.config.set(
        {
            "codec_pipeline.path": "zarrs.ZarrsCodecPipeline",
            "threading.max_workers": max(len(os.sched_getaffinity(0)) // (world * args.workers), 1),
        }
    )
    dist_info = "torch" if world > 1 else (lambda: (0, 1))
    results = OUT_ROOT / "results"

    def make_loader(split: str, shuffle: bool) -> torch.utils.data.DataLoader:
        """The stream of a split for this rank: ``global_batch // world`` spots per batch, a fixed shard of the spots."""
        sampler_args = {
            "chunk_size": args.chunk_size,
            "preload_nchunks": args.preload_nchunks,
            "batch_size": args.global_batch // world,
            "drop_last": True,
        }
        sampler = DistributedSampler(
            RandomSampler(**sampler_args, rng=np.random.default_rng(0))
            if shuffle
            else SequentialSampler(**sampler_args),
            dist_info=dist_info,
        )
        collection = SpatialDatasetCollection(zarr.open_group(collection_path(split), mode="r"))
        loader = Loader(
            images=[HE_ALIGNED], return_annotations=["sample_id", "instance_id"], batch_sampler=sampler, to="torch"
        ).use_collection(collection)
        return torch.utils.data.DataLoader(
            loader,
            batch_size=None,
            num_workers=args.workers if shuffle else 1,  # a sequential sampler does not support several workers
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2,
        )

    margin = (TILE_PX - MODEL_PX) // 2  # the tile is centered on its spot, so a center crop keeps the spot centered

    def prepare(batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Model inputs on the GPU: the tile as ``(batch, 3, 224, 224)`` and log-normalised expression ``(batch, genes)``."""
        tiles = batch["images"][HE_ALIGNED].to(device, non_blocking=True)  # (batch, 3, 512, 512) uint8
        tiles = (
            tiles[..., margin : margin + MODEL_PX, margin : margin + MODEL_PX]
            if args.input == "crop"
            else nn.functional.interpolate(tiles.float(), size=MODEL_PX, mode="bilinear", antialias=True)
        ).float()
        tiles = torch.log1p(tiles)
        tiles = (tiles - tiles.mean(dim=(-2, -1), keepdim=True)) / tiles.std(dim=(-2, -1), keepdim=True).clamp(min=1e-3)
        x = batch["X"].to(device, non_blocking=True).to_dense()
        return tiles, torch.log1p(x / x.sum(dim=1, keepdim=True).clamp(min=1) * 1e4)

    train_loader, val_loader = make_loader("train", shuffle=True), make_loader("val", shuffle=False)
    first = next(iter(train_loader))
    n_genes = first["X"].shape[1]

    if rank == 0:  # what a batch looks like, and that a tile is centered on its spot
        tile = first["images"][HE_ALIGNED]
        assert tuple(tile.shape[1:]) == (3, TILE_PX, TILE_PX) and tile.dtype == torch.uint8, tile.shape
        print({k: tuple(v.shape) for k, v in first["images"].items()}, "X", tuple(first["X"].shape), flush=True)
        store = slides()["train"][0]
        spot = sd.read_zarr(store, selection=("shapes",))[REGION].geometry.iloc[0]
        vertices = (
            np.asarray(spot.exterior.coords) - np.asarray(spot.centroid.coords)
        ) * 4.70588235  # um -> px of global
        genes = json.loads((results / f"collection_train{SUFFIX}.json").read_text())["genes"]
        fig, axes = plt.subplots(2, 4, figsize=(14, 7.4))
        for ax, row in zip(axes.ravel(), range(8), strict=True):
            ax.imshow(tile[row].permute(1, 2, 0).numpy())
            ax.plot(*(vertices + TILE_PX / 2).T, "c-", lw=1)
            top = first["X"][row].to_dense().argsort(descending=True)[:3]
            ax.set_title(
                f"{first['obs']['sample_id'].iloc[row][-22:]}\n{', '.join(genes[i] for i in top.tolist())}",
                fontsize=7,
            )
            ax.axis("off")
        fig.suptitle("512 px H&E tiles (108.8 µm), their 55 µm spot outline, and the top genes of the spot")
        fig.savefig(results / f"tiles_with_spots{SUFFIX}.png", dpi=130, bbox_inches="tight")

    model = nn.Sequential(
        nn.Conv2d(3, 16, 5, stride=2, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, n_genes),
    ).to(device)
    if world > 1:
        model = DistributedDataParallel(model, device_ids=[device.index])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for step, batch in enumerate(train_loader):
        tiles, target = prepare(batch)
        loss = nn.functional.mse_loss(model(tiles), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if rank == 0 and step % 25 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}", flush=True)
        if step + 1 == args.steps:
            break

    record = {"input": args.input, "world_size": world, "global_batch": args.global_batch, "train_loss": losses}
    if rank == 0:  # kept even if the validation below fails
        (results / f"tutorial_{args.input}{SUFFIX}.json").write_text(json.dumps({**record, "val_mse": None}))

    model.eval()
    val_losses = []
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            tiles, target = prepare(batch)
            val_losses.append(nn.functional.mse_loss(model(tiles), target).item())
            if step + 1 == args.val_steps:
                break
    val_mse = torch.tensor(float(np.mean(val_losses)), device=device)
    if world > 1:
        dist.all_reduce(val_mse, op=dist.ReduceOp.AVG)
    if rank == 0:
        print(
            f"val MSE {val_mse.item():.3f} (train: first 10 steps {np.mean(losses[:10]):.3f} -> last 10 {np.mean(losses[-10:]):.3f})"
        )
        (results / f"tutorial_{args.input}{SUFFIX}.json").write_text(json.dumps({**record, "val_mse": val_mse.item()}))
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
