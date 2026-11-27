# HESCAPE multi-tissue collections

Writes the flagship multi-tissue Xenium stores (`/p/scratch/hai_1240/datasets/flagship/spatialdata_zarr/multi_tissue`) as three
`spatialbatch` collections of paired H&E tiles and 55 µm spot expression, and tests the multimodal tutorial on them.

| step | script                                           | what it does                                                                                                                        |
| ---- | ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1    | `align_he.py`, `slurm/align.sbatch`              | resamples `he_image` onto the pixel grid of `global` as `he_image_aligned`, in the same store (job array, one slide per task)       |
| 2    | `build_collections.py`, `slurm/build.sbatch`     | writes `train`, `val` and `test` as `SpatialDatasetCollection`s: in-tissue spots only, the genes shared by every slide in one order |
| 3    | `verify_collections.py`                          | checks rows, genes, tile size, split disjointness and tile pixels against the source stores                                         |
| 4    | `multimodal_hescape.py`, `slurm/tutorial.sbatch` | the multimodal tutorial with DistributedDataParallel at a global batch of 1024                                                      |
| 5    | `benchmark_loader.py`, `slurm/bench*.sbatch`     | loader throughput on 1, 2 and 4 GPUs of a node                                                                                      |
| 6    | `profile_request.py`, `simulate_order.py`        | where one request spends its time, and bytes read per spot for other row orders                                                     |
| 7    | `make_report.py`                                 | builds `results/report.html` from the files above                                                                                   |

`slides.py` chooses the slides (those with an `he_image`, minus `EXCLUDED`), the split and the write parameters. Everything is written under
`$HESCAPE_OUT` (default `/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue`); `HESCAPE_SUFFIX=_v2` writes a second version of
the collections (and of their result files) next to the first. Jobs use `--partition=dc-hwai --account=hai_1240`; submit them from the repository
root. See the report for the commands in order and for what the measurements show.
