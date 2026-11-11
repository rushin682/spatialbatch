# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- `Loader`, an `annbatch.Loader` that streams cell tiles of images and labels with the expression and annotations of
  the cell from SpatialData stores, reproducing `spatialstream`'s `XeniumDataset`.
- `RandomSampler`, built on `annbatch.samplers.RandomSampler` with defaults for pixel tiles.
- `SpatialDatasetCollection` and `write_sharded_sdata`: spatially sorted (Morton order), sharded copies of the stores.
- `write_sharded_sdata` resamples images and labels with any affine transformation (rotation, scale, shear,
  sub-pixel shift) into a common pixel grid, with the nearest-neighbour sampling and pyramid level of
  `spatialdata.rasterize`, and copies elements shifted by whole pixels as they are. The grid resolution is set with
  `target_unit_to_pixels`. Collection encoding 0.2.0 records each array's origin; 0.1.0 collections still load.
