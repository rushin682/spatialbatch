# Quickstart

{class}`spatialbatch.Loader` streams minibatches out of on-disk {class}`~spatialdata.SpatialData` stores:
for every cell, a tile of each requested image and labels element together with the cell's expression and annotations.

```python
import spatialdata as sd
from spatialbatch import Loader

loader = Loader(
    images=["morphology_focus"],
    labels=["cell_labels"],
    tile_dim_in_units=224,
    batch_size=32,
    shuffle=True,
    to="torch",
).add_sdata(sd.read_zarr("xenium.zarr"))

for batch in loader:
    batch["images"]["morphology_focus"]  # (32, 1, 224, 224)
    batch["X"]  # (32, n_genes) sparse expression
```

For the best throughput, write the stores once into a {class}`~spatialbatch.SpatialDatasetCollection`
and see {doc}`zarr-configuration`. The {doc}`API reference <api>` lists every class and function.

## Images with their own transformation

Tiles are read by slicing only from elements in the pixel grid of the coordinate system (their transformation to it
is the identity). Others, such as an H&E image registered to the slide with a rotation or a scaling, are tiled with
one spatialdata call per tile, which is much slower, and cannot be written to a collection. Resample them once with
{func}`spatialbatch.align_to_cs`:

```python
from spatialbatch import align_to_cs

sdata = align_to_cs("xenium.zarr", "he_image")  # adds `he_image_aligned` to the store
# or: align_to_cs("xenium.zarr", "he_image", out_path="xenium_aligned.zarr")  # a copy, `he_image` replaced
loader = Loader(images=["morphology_focus", "he_image_aligned"], tile_dim_in_units=224).add_sdata(sdata)
```

Pass `like="morphology_focus"` to give the aligned image the extent of the morphology image.
