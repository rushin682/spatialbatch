"""The abstract sampler is shared with annbatch: a sampler written for one loader works in the other."""

from __future__ import annotations

from annbatch.abc.sampler import Sampler

__all__ = ["Sampler"]
