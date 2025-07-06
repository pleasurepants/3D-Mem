#!/bin/bash
#SBATCH --job-name=q-po-o-o
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/qwen/point-only-only-%j.out 
#SBATCH --partition a40



# srun --nodes=1 --gres=gpu:a100:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a40 --pty bash

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
echo "=== JOB START ==="
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"


if [ -z "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[INFO] SLURM_JOB_GPUS not set, fallback to 0,1"
else
    export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
    echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

echo "[INFO] Starting vLLM (qwen) server on GPU 0..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate vllm

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \

# vllm serve /anvme/workspace/v100dd12-3dmem/model/MiniCPM-V-2_6 \
#     --served-model-name minicpm \
#     --port 8000 \
#     --limit-mm-per-prompt image=20 \
#     --trust-remote-code &

vllm serve /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-VL-3B-Instruct \
    --served-model-name qwen \
    --port 8000 \
    --limit-mm-per-prompt image=20 &
VLLM_PID=$!


echo "[INFO] Waiting for vLLM (qwen) server to be ready..."
for i in {1..300}; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "[INFO] ✅ qwen API is ready!"
        break
    fi
    echo "  ... waiting ($((i*10))s)"
    sleep 10
    if [ $i -eq 300 ]; then
        echo "[ERROR] ❌ Timeout: qwen server failed to start."
        if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
            kill "$VLLM_PID"
        fi
        exit 1
    fi
done


echo "[INFO] Starting AEQA evaluation on GPU 1 (3dmem env)..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source .env
# export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/cfg/alex_cfg/qwen_only.yaml


echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="
