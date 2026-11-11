"""Optional-dependency shims (torch, cupy, jax), shared with annbatch."""

from __future__ import annotations

from annbatch.compat import CupyArray, CupyCSRMatrix, IterableDataset, JaxArray, JAXCSRMatrix, Tensor

__all__ = ["CupyArray", "CupyCSRMatrix", "IterableDataset", "JAXCSRMatrix", "JaxArray", "Tensor"]
