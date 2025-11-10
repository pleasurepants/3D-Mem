#!/bin/bash
#SBATCH --job-name=gpt-hiera-cot-44
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/gpt/hiera-cot-44-%j.out 
#SBATCH --partition a40



# srun --nodes=1 --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:1 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a40 --pty bash

# Set proxy for accessing OpenAI API
export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
echo "=== JOB START ==="
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

echo "[INFO] Starting AEQA evaluation(3dmem env)..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source .env

# OPENAI_API_KEY is loaded from .env file
MASTER_ADDR=localhost
#   -m debugpy --listen 0.0.0.0:8798 --wait-for-client 
# export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_gpt.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/cfg/alex_cfg/gpt-hiera-cot-44.yaml \
    --base_mode hierarchy \
    --chat_seed 44


echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="
