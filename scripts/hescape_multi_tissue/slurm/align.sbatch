#!/bin/bash
# Align the H&E image of every slide to `global`, one slide per array task: sbatch --array=0-25 slurm/align.sbatch
# Submit from the repository root.
#SBATCH --job-name=hescape-align
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=110G
#SBATCH --time=06:00:00
#SBATCH --output=/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/logs/align_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
.venv/bin/python scripts/hescape_multi_tissue/align_he.py "${SLURM_ARRAY_TASK_ID}" --workers "${SLURM_CPUS_PER_TASK}"
