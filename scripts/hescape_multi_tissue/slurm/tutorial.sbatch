#!/bin/bash
# The multimodal tutorial on the collections: global batch 1024 over the 4 GPUs of a node, once with a 224 center crop and
# once with a resize of the 512 px tile. Needs train.zarr and val.zarr. Submit from the repository root.
#SBATCH --job-name=hescape-tutorial
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64   # 2 threads per core: this is the whole node, 128 processors
#SBATCH --mem=480G
#SBATCH --time=01:00:00
#SBATCH --output=/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/logs/tutorial_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
for input in resize crop; do
  .venv/bin/torchrun --standalone --nproc-per-node 4 scripts/hescape_multi_tissue/multimodal_hescape.py --input "${input}"
done
