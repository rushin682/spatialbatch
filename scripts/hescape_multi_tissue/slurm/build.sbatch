#!/bin/bash
# Write one collection: sbatch slurm/build.sbatch <train|val|test> [--overwrite]. Submit from the repository root,
# after every slide of the split is aligned (see align.sbatch).
#SBATCH --job-name=hescape-build
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=12:00:00
#SBATCH --output=/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/logs/build_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
.venv/bin/python scripts/hescape_multi_tissue/build_collections.py "$@"
