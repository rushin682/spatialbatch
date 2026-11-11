"""RandomSampler -- shuffled chunk-based sampler."""

from __future__ import annotations

from typing import TYPE_CHECKING

from annbatch.samplers import RandomSampler as _AnnbatchRandomSampler

if TYPE_CHECKING:
    import numpy as np


class RandomSampler(_AnnbatchRandomSampler):
    """Shuffled chunk-based sampler for batched tile access.

    This is :class:`annbatch.samplers.RandomSampler` (and thereby :class:`annbatch.abc.Sampler`) unchanged in behaviour:
    a *chunk* is a run of `chunk_size` consecutive loader rows (cells), chunks are drawn in random order, and
    `preload_nchunks` chunks are fetched together before being shuffled in memory and split into batches.
    Worker sharding and per-worker RNG spawning are inherited.

    What differs are the defaults, which are tuned for pixels rather than expression rows.
    A cell tile is ~100 KB (one 224x224 uint16 channel), about two orders of magnitude heavier than a sparse expression row,
    so one request defaults to ``64 * 8 = 512`` tiles instead of ``512 * 32`` rows.
    Chunks are most effective when consecutive rows are spatially close, which
    :class:`~spatialbatch.SpatialDatasetCollection` guarantees by storing cells in Morton (Z-order) order.

    Parameters
    ----------
    batch_size
        Number of observations per batch.
    chunk_size
        Number of consecutive cells per chunk.
    preload_nchunks
        Number of chunks to load per iteration.
    mask
        A slice defining the observation range to sample from (start:stop).
    drop_last
        Whether to drop the last incomplete batch.
    rng
        Random number generator for shuffling, a seeded :class:`numpy.random.Generator` for reproducibility.
    replacement
        If ``True``, draw random chunks with replacement.
    num_samples
        Total number of observations to draw, see :class:`annbatch.samplers.RandomSampler`.
    """

    def __init__(
        self,
        chunk_size: int = 64,
        preload_nchunks: int = 8,
        batch_size: int = 32,
        *,
        replacement: bool = False,
        num_samples: int | None = None,
        drop_last: bool = False,
        mask: slice | None = None,
        rng: np.random.Generator | None = None,
    ):
        super().__init__(
            chunk_size=chunk_size,
            preload_nchunks=preload_nchunks,
            batch_size=batch_size,
            replacement=replacement,
            num_samples=num_samples,
            drop_last=drop_last,
            mask=mask,
            rng=rng,
        )
