#!/bin/bash
#SBATCH --job-name=perplexity-score
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/perplexity/perplexity-%j.out
#SBATCH --partition a40


# srun --nodes=1 --gres=gpu:a100:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:1 --ntasks=1 --cpus-per-task=16 --time=1:00:00 --partition a40 --pty bash

# Set paths
JSON_FILE="/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/traj_abs_format.json"
OUTPUT_FILE="/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/traj_abs_format_per.json"

# Check if file exists
if [ ! -f "$JSON_FILE" ]; then
    echo "Error: File does not exist: $JSON_FILE"
    exit 1
fi

echo "=== JOB START ==="
echo "JSON file path: $JSON_FILE"
echo "Output file path: $OUTPUT_FILE"
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Set environment variables
export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

if [ -z "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "[INFO] SLURM_JOB_GPUS not set, fallback to 0"
else
    export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
    echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

# Activate conda environment
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Run Python script
echo "=== Starting Perplexity Calculation ==="
python -m debugpy --listen 0.0.0.0:8798 --wait-for-client /home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/perpexity_score.py \
    --json_dir "$JSON_FILE" \
    --output_dir "$OUTPUT_FILE" \
    --model_name /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-7B \
    --max_length 2048 \
    --stride 1024

echo "=== JOB END ==="
date
