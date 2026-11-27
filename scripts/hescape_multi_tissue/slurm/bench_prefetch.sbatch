#!/bin/bash
# The Loader with and without prefetch on 1, 2 and 4 GPUs, with a cold page cache for the first run and a warm one for
# the rest: HESCAPE_SUFFIX=_v2 sbatch slurm/bench_prefetch.sbatch. Submit from the repository root.
#SBATCH --job-name=hescape-bench2
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64   # 2 threads per core: this is the whole node, 128 processors
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/logs/bench_prefetch_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
BENCH=scripts/hescape_multi_tissue/benchmark_loader.py
COLLECTION=/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/train${HESCAPE_SUFFIX:-}.zarr
run() {  # gpus, prefetch flag, step_ms, tag
  .venv/bin/torchrun --standalone --nproc-per-node "$1" "${BENCH}" train --mode bare --preload-nchunks "$(( $1 == 1 ? 256 : 128 ))" \
    "$2" --step-ms "$3" --tag "$4"
}

# a node whose page cache has not seen the collection: the first epoch
run 4 --no-prefetch 0 cold
run 4 --prefetch 0 cold
# the collection is a fraction of the RAM: read it once so that the following runs are the later epochs
find "${COLLECTION}" -type f -print0 | xargs -0 -P 32 cat > /dev/null
for step_ms in 0 300; do
  for gpus in 4 2 1; do
    for flag in --no-prefetch --prefetch; do
      run "${gpus}" "${flag}" "${step_ms}" warm
    done
  done
done
