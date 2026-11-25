from importlib.metadata import version

from . import abc, samplers, types
from .align import align_to_cs
from .io import SpatialDatasetCollection, write_sharded_sdata
from .loader import Loader

__version__ = version("spatialbatch")

__all__ = ["Loader", "SpatialDatasetCollection", "abc", "align_to_cs", "samplers", "types", "write_sharded_sdata"]
