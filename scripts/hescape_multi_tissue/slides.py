"""Which slides of the flagship multi-tissue dataset go to which collection (shared by every script of the pipeline)."""

from __future__ import annotations

import os
from pathlib import Path

#: The flagship SpatialData stores: one folder per cohort, one ``.zarr`` per slide.
SDATA_ROOT = Path(
    os.environ.get("HESCAPE_SDATA", "/p/scratch/hai_1240/datasets/flagship/spatialdata_zarr/multi_tissue")
)
#: Where the collections, logs and results are written.
OUT_ROOT = Path(os.environ.get("HESCAPE_OUT", "/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue"))

#: Appended to the name of a collection (``train<suffix>.zarr``) and of its result files, to build another version of
#: the collections next to the first: ``HESCAPE_SUFFIX=_v2``.
SUFFIX = os.environ.get("HESCAPE_SUFFIX", "")

#: The cohort used for training and validation: every other cohort is the test set.
TRAIN_COHORT = "TENX"
#: TENX slides held out as validation, chosen at slide level (no tile of a spot overlaps another slide) with a sibling
#: slide of the same tissue left in train: 64.3k of the 319k spots in tissue, 20.1%.
VAL_SLIDES = frozenset(
    {
        "Xenium_V1_hKidney_nondiseased_section",
        "Xenium_V1_hLiver_nondiseased_section_FFPE",
        "Xenium_V1_hSkin_nondiseased_section_2_FFPE",
    }
)
#: Slides that are left out although they have an ``he_image``, with the reason.
EXCLUDED = {
    "UKER/MIER_226C-bl_B": "its he_image -> global transformation is the identity: the H&E is not registered",
}

#: The element aligned to the pixel grid of ``global``, the element that gives its extent, and the aligned name.
HE_IMAGE, LIKE, HE_ALIGNED = "he_image", "cell_labels", "he_image_aligned"
REGION, TABLE = "spots_55um", "spots_55um_table"
#: Tile side in pixels of ``global`` (0.2125 um/px): 108.8 um, so that a 224 crop or a resize to 224 is possible.
TILE_PX = 512
#: Side in pixels of the grid the rows are ordered on (the window the loader reads), and of the zarr chunks and shards of
#: the image. A shard holds 32 x 32 chunks of 3 x 128 x 128 bytes: about 50 MB before compression.
BLOCK_PX, IMAGE_CHUNK, IMAGE_SHARD = 1024, 128, 4096


def collection_path(split: str) -> Path:
    """Where the collection of a split is written."""
    return OUT_ROOT / f"{split}{SUFFIX}.zarr"


def slides() -> dict[str, list[Path]]:
    """The stores of each split: slides without an ``he_image`` and the excluded ones are ignored."""
    splits: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for cohort in sorted(p for p in SDATA_ROOT.iterdir() if p.is_dir()):
        for store in sorted(cohort.glob("*.zarr")):
            if not (store / "images" / HE_IMAGE).exists() or f"{cohort.name}/{store.stem}" in EXCLUDED:
                continue
            if cohort.name != TRAIN_COHORT:
                splits["test"].append(store)
            else:
                splits["val" if store.stem in VAL_SLIDES else "train"].append(store)
    return splits
