# Quickstart

spatialbatch works on {class}`~anndata.AnnData` objects: {mod}`spatialbatch.pp` preprocesses them, {mod}`spatialbatch.tl` computes on them and {mod}`spatialbatch.pl` plots them.

```python
import numpy as np
from anndata import AnnData

import spatialbatch

adata = AnnData(np.random.normal(size=(20, 10)))

spatialbatch.pp.basic_preproc(adata)
spatialbatch.tl.basic_tool(adata)
spatialbatch.pl.basic_plot(adata)
```

The {doc}`tutorials <notebooks/example>` go into more detail and the {doc}`API reference <api>` lists every function.
