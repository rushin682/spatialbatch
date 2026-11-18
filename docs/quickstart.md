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
