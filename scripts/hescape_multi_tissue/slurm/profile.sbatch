#!/bin/bash
# Per-request profile of the loader on one node (CPU only): sbatch slurm/profile.sbatch <split> [profile_request.py args].
# Submit from the repository root.
#SBATCH --job-name=hescape-profile
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64   # 2 threads per core: this is the whole node, 128 processors
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/logs/profile_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
.venv/bin/python scripts/hescape_multi_tissue/profile_request.py "$@"
