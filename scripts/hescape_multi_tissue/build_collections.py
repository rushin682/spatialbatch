"""Write one split (train, val or test) as a SpatialDatasetCollection of ``he_image_aligned`` + ``spots_55um_table``.

Only the spots in tissue are kept, and every table is reduced to the same genes in the same order: the loader checks the
number of genes of the datasets, not their names, so a different order would be silently wrong.

Usage: ``build_collections.py <train|val|test> [--overwrite]``
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import warnings
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import spatialdata as sd
import zarr
from slides import (
    BLOCK_PX,
    HE_ALIGNED,
    IMAGE_CHUNK,
    IMAGE_SHARD,
    OUT_ROOT,
    REGION,
    SUFFIX,
    TABLE,
    TILE_PX,
    collection_path,
    slides,
)

from spatialbatch import SpatialDatasetCollection

if TYPE_CHECKING:
    from pathlib import Path

warnings.filterwarnings("ignore", message="Consolidated metadata is currently not part in the Zarr format 3.*")
warnings.filterwarnings("ignore", message="The index of the dataframe is not monotonic increasing.*")


def gene_panel(stores: list[Path]) -> list[str]:
    """The genes shared by every slide of every split, in the order of the first slide."""
    panels = [ad.io.read_elem(zarr.open_group(path, mode="r")[f"tables/{TABLE}/var"]).index for path in stores]
    shared = set(panels[0]).intersection(*map(set, panels[1:]))
    return [gene for gene in panels[0] if gene in shared]


def in_tissue(path: Path) -> sd.SpatialData:
    """The slide, backed by its store, with the spots (shapes and table) outside the tissue removed."""
    sdata = sd.read_zarr(path)
    table = sdata.tables[TABLE]
    kept = table.obs["in_tissue"].to_numpy() == 1
    table = table[kept].copy()
    table.obs["sample_id"] = path.stem
    table.obs["cohort"] = path.parent.name
    spots = sdata[REGION]
    del sdata[REGION]
    sdata[REGION] = spots[spots.index.isin(table.obs["instance_id"])]  # keeps the transformations of the shapes
    sdata.tables[TABLE] = table
    return sdata


parser = argparse.ArgumentParser()
parser.add_argument("split", choices=["train", "val", "test"])
parser.add_argument("--overwrite", action="store_true")
args = parser.parse_args()

zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline", "threading.max_workers": os.cpu_count()})
every_store = [path for split in slides().values() for path in split]
genes = gene_panel(every_store)
stores = slides()[args.split]
out = collection_path(args.split)
if out.exists() and any(out.glob("dataset_*")):  # an empty group is what a failed run leaves: it is just rewritten
    if not args.overwrite:
        raise SystemExit(f"{out} holds a collection already; pass --overwrite to replace it.")
    shutil.rmtree(out)
print(f"{args.split}: {len(stores)} slides, {len(genes)} genes -> {out}", flush=True)


def load_table(table: ad.AnnData) -> ad.AnnData:
    """What is stored: raw counts as float32 for the shared genes, three annotations and the spot coordinates."""
    table = table[:, genes]
    return ad.AnnData(
        X=table.X.astype(np.float32),
        obs=table.obs[["instance_id", "sample_id", "cohort"]],
        var=table.var,
        obsm={"spatial": table.obsm["spatial"]},
    )


sdatas = [in_tissue(path) for path in stores]  # before the collection exists, so a failure leaves nothing behind
collection = SpatialDatasetCollection(zarr.open_group(out, mode="w"))
collection.add_sdatas(
    sdatas,
    region=REGION,
    table_name=TABLE,
    images=[HE_ALIGNED],
    tile_dim_in_units=TILE_PX,
    block_px=BLOCK_PX,
    image_chunk=IMAGE_CHUNK,
    image_shard=IMAGE_SHARD,
    load_table=load_table,
)

datasets = [
    {
        "name": group.name,
        "n_spots": int(group["obsm"]["spatial"].shape[0]),
        **group.attrs["spatialbatch"],
    }
    for group in collection
]
size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
(OUT_ROOT / "results" / f"collection_{args.split}{SUFFIX}.json").write_text(
    json.dumps({"split": args.split, "genes": genes, "bytes": size, "datasets": datasets}, indent=1)
)
print(f"{args.split}: {sum(d['n_spots'] for d in datasets):,} spots, {size / 1e9:.1f} GB on disk", flush=True)
