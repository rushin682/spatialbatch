# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/
[semantic versioning]: https://semver.org/

## [Unreleased]

### Added

- Tutorials: a quickstart and guides to stream the table, the images and labels, and both together from one collection.
- `Loader(table=False)` and empty `images` / `labels`: one collection written with all elements can be mini-batched
  for any subset of them (table only, images only, labels only...). Elements that are not requested are never read.
- `write_sharded_sdata` / `SpatialDatasetCollection.add_sdatas`: `load_table`, a callback that chooses what of the table
  is written (like `load_adata` in annbatch), e.g. to write no expression. `images` and `labels` may be empty, for a
  table-only collection.

### Changed

- Tiles are 1024 x 1024 units by default (`tile_dim_in_units=1024`, pixels for Xenium), and collections store images
  and labels in 128 x 128 pixel chunks, 8 per tile side. Crop to the size a model needs in `transform`.
  Pass `tile_dim_in_units=None` to size tiles from the cell diameter with `tile_scale`, as before.
- `Loader` and `RandomSampler` default to `chunk_size=16` (was 64), as a default tile is ~20x heavier.

### Fixed

- Writing an image smaller than a shard whose side is not a multiple of `image_chunk`.

- `align_to_cs`: resample images and labels whose transformation to a coordinate system is not the identity (e.g. a
  registered H&E image) onto its pixel grid with `spatialdata.rasterize`, once, so that they are tiled by slicing and
  can be written to a `SpatialDatasetCollection`.

## [0.0.1rc1] - 2026-11-20

### Added

- `Loader`, an `annbatch.Loader` that streams cell tiles of images and labels with the expression and annotations of
  the cell from SpatialData stores, reproducing `spatialstream`'s `XeniumDataset`.
- `RandomSampler`, built on `annbatch.samplers.RandomSampler` with defaults for pixel tiles.
- `SpatialDatasetCollection` and `write_sharded_sdata`: spatially sorted (Morton order), sharded copies of the stores.
