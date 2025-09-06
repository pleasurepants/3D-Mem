#!/bin/bash
#SBATCH --job-name=frontier-ahash-index
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/ee/frontier-ahash-index-%j.out
#SBATCH --partition a100

set -eo pipefail

ROOT=${1:-/anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168}
WORKERS=${WORKERS:-16}
HASH_SIZE=${HASH_SIZE:-8}

source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem

echo "[IndexJob] root=$ROOT workers=$WORKERS hash=$HASH_SIZE"; date; hostname
nvidia-smi || true
python /home/hpc/v100dd/v100dd12/code/3D-Mem/cursor/build_frontier_ahash_index.py \
  --root "$ROOT" \
  --workers "$WORKERS" \
  --hash_size "$HASH_SIZE" \
  --merge | cat

echo "[IndexJob] done."


