"""Check the written collections against their source stores, and write ``results/verification.json``.

For every split: the rows of each dataset are the spots in tissue of its slide, the genes are the shared panel in one
order, every tile is ``TILE_PX`` wide and centered on its spot, tiles read through the Loader are the pixels of the
source ``he_image_aligned`` at their origin, and no slide is in two splits.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import zarr
from slides import HE_ALIGNED, OUT_ROOT, SUFFIX, TILE_PX, collection_path, slides
from zarr import Group

from spatialbatch import Loader, SpatialDatasetCollection
from spatialbatch.utils import scale0

warnings.filterwarnings("ignore")
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
PX_PER_UM = 4.70588235  # the scale of spots_55um to global
N_TILES = 16

report: dict[str, dict] = {}
seen: dict[str, str] = {}
for split, stores in slides().items():
    collection = SpatialDatasetCollection(zarr.open_group(collection_path(split), mode="r"))
    panel = json.loads((OUT_ROOT / "results" / f"collection_{split}{SUFFIX}.json").read_text())["genes"]
    groups: list[Group] = list(collection)
    assert len(groups) == len(stores), (split, len(groups), len(stores))
    datasets = []
    for group, store in zip(groups, stores, strict=True):
        meta = group.attrs["spatialbatch"]
        table = zarr.open_group(store, mode="r")["tables/spots_55um_table"]
        in_tissue = int((np.asarray(table["obs"]["in_tissue"][:]) == 1).sum())
        spots = np.asarray(group["obsm"]["spatial"][:]) * PX_PER_UM
        offset = np.abs(np.asarray(group["obsm"]["tile_origin"][:]) + TILE_PX // 2 - spots).max()
        assert meta["source"] == str(store) and meta["tile_hw"] == [TILE_PX, TILE_PX], meta
        assert spots.shape[0] == in_tissue, (store.stem, spots.shape[0], in_tissue)
        assert offset <= 0.51, (store.stem, offset)  # tile start is the rounded corner: half a pixel at most
        assert store.stem not in seen, f"{store.stem} is in {seen[store.stem]} and {split}"
        seen[store.stem] = split
        datasets.append(
            {
                "slide": store.stem,
                "cohort": store.parent.name,
                "n_spots": in_tissue,
                "max_center_offset_px": float(offset),
            }
        )
    loader = Loader(
        images=[HE_ALIGNED],
        return_index=True,
        batch_size=N_TILES,
        chunk_size=32,
        preload_nchunks=8,
        shuffle=True,
        rng=np.random.default_rng(0),
    ).use_collection(collection)
    assert list(loader.var.index) == panel, split
    batch = next(iter(loader))
    bounds = np.cumsum([0] + [d["n_spots"] for d in datasets])
    equal = 0
    for row, index in enumerate(batch["index"].tolist()):
        which = int(np.searchsorted(bounds, index, side="right") - 1)
        x0, y0 = np.asarray(groups[which]["obsm"]["tile_origin"][index - bounds[which]]).tolist()
        array = scale0(str(stores[which]), "images", HE_ALIGNED)
        expected = np.zeros((3, TILE_PX, TILE_PX), np.uint8)
        height, width = array.shape[-2:]
        y_lo, y_hi, x_lo, x_hi = max(y0, 0), min(y0 + TILE_PX, height), max(x0, 0), min(x0 + TILE_PX, width)
        expected[:, y_lo - y0 : y_hi - y0, x_lo - x0 : x_hi - x0] = array[:, y_lo:y_hi, x_lo:x_hi]
        equal += int(np.array_equal(np.asarray(batch["images"][HE_ALIGNED][row]), expected))
    assert equal == N_TILES, (split, equal)
    report[split] = {"n_spots": int(bounds[-1]), "n_genes": len(panel), "tiles_checked": N_TILES, "datasets": datasets}
    print(
        f"{split}: {len(datasets)} slides, {bounds[-1]:,} spots, {len(panel)} genes, {equal}/{N_TILES} tiles pixel-exact",
        flush=True,
    )
(OUT_ROOT / "results" / f"verification{SUFFIX}.json").write_text(json.dumps(report, indent=1))
