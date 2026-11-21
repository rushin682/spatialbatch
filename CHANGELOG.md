# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `align_to_cs`: resample images and labels whose transformation to a coordinate system is not the identity (e.g. a
  registered H&E image) onto its pixel grid with `spatialdata.rasterize`, once, so that they are tiled by slicing and
  can be written to a `SpatialDatasetCollection`.

## [0.0.1rc1] - 2026-11-20

### Added

- `Loader`, an `annbatch.Loader` that streams cell tiles of images and labels with the expression and annotations of
  the cell from SpatialData stores, reproducing `spatialstream`'s `XeniumDataset`.
- `RandomSampler`, built on `annbatch.samplers.RandomSampler` with defaults for pixel tiles.
- `SpatialDatasetCollection` and `write_sharded_sdata`: spatially sorted (Morton order), sharded copies of the stores.
