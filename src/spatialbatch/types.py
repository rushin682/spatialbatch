from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from annbatch.types import BackingArray_T, LoaderOutput, LoadRequest, OutputInMemoryArray_T

from .compat import JaxArray, Tensor

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "BackingArray_T",
    "LoadRequest",
    "LoaderOutput",
    "OutputInMemoryArray_T",
    "SpatialLoaderOutput",
    "TileArray_T",
]

type TileArray_T = np.ndarray | Tensor | JaxArray


class SpatialLoaderOutput(TypedDict):
    """The output of :class:`spatialbatch.Loader`: one batch of cells with every requested modality.

    It carries the keys of :class:`annbatch.types.LoaderOutput` (``X``, ``obs``, ``var``, ``index``) plus the pixel
    modalities, all aligned along axis 0 (the batch).

    Attributes
    ----------
    X
        The expression matrix of the batch, ``(batch, n_var)``, sparse unless the loader densifies it.
    obs
        The ``obs`` rows of the batch, or ``None`` when the table has no ``obs`` columns.
    var
        The ``var`` annotations.
    index
        Position of each row in the loader (across all added SpatialData objects), or ``None``.
    images
        ``{image name: (batch, channels, height, width)}``.
    labels
        ``{labels name: (batch, height, width)}``.
    coords
        ``obsm["spatial"]`` of the batch, ``(batch, 2)``, or ``None``.
    """

    X: OutputInMemoryArray_T.__value__
    obs: pd.DataFrame | None
    var: pd.DataFrame | None
    index: np.ndarray | None
    images: dict[str, TileArray_T]
    labels: dict[str, TileArray_T]
    coords: np.ndarray | None
