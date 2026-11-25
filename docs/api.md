# API

```{eval-rst}
.. module:: spatialbatch
.. currentmodule:: spatialbatch
```

## Loader

```{eval-rst}
.. autosummary::
    :toctree: generated

    Loader
```

## Collections

```{eval-rst}
.. autosummary::
    :toctree: generated

    SpatialDatasetCollection
    write_sharded_sdata
```

## Alignment

```{eval-rst}
.. autosummary::
    :toctree: generated

    align_to_cs
```

## Transforms

Helpers for the `transform` of a {class}`~spatialbatch.Loader`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    center_crop
```

## Samplers

The abstract {class}`annbatch.abc.Sampler` is shared with annbatch, as is {class}`annbatch.samplers.SequentialSampler`.

```{eval-rst}
.. autosummary::
    :toctree: generated

    samplers.RandomSampler
    abc.Sampler
```

## Types

```{eval-rst}
.. autosummary::
    :toctree: generated
    :template: typeddict.rst

    types.SpatialLoaderOutput
```
