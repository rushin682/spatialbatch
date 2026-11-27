#!/bin/bash
# Loader throughput on 1, 2 and 4 GPUs of one node, bare Loader vs. torch DataLoader workers, written to
# results/benchmark.jsonl. Submit from the repository root once train.zarr exists.
#SBATCH --job-name=hescape-bench
#SBATCH --account=hai_1240
#SBATCH --partition=dc-hwai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64   # 2 threads per core: this is the whole node, 128 processors
#SBATCH --mem=480G
#SBATCH --time=03:00:00
#SBATCH --output=/p/scratch/hai_1240/datasets/spatialbatch_testing/hescape/multi_tissue/logs/bench_%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
BENCH=scripts/hescape_multi_tissue/benchmark_loader.py
for step_ms in 0 300; do
  for gpus in 4 2 1; do
    # chunks of 32 spots: 128 chunks are a request of 4096 spots, and 256 on one GPU, whose batch is the global one
    preload=$((gpus == 1 ? 256 : 128))
    for mode in bare workers; do
      .venv/bin/torchrun --standalone --nproc-per-node "${gpus}" "${BENCH}" train \
        --mode "${mode}" --preload-nchunks "${preload}" --step-ms "${step_ms}" --tag "main"
    done
  done
done
