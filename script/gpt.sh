#!/bin/bash
#SBATCH --job-name=internvl_bl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --output=/home/wiss/zhang/code/openeqa/3D-Mem/slurm/baseline/internvl_baseline-%j.out
#SBATCH --partition all

# echo "=== JOB START ==="
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
# srun --pty --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:2 --time=21:00:00 --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-8,worker-9,worker-1,worker-2,worker-3,worker-4,worker-6 --partition all bash

# srun --pty --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --time=12:00:00 --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-8,worker-9 --partition all bash


# if [ -z "$SLURM_JOB_GPUS" ]; then
#     export CUDA_VISIBLE_DEVICES=0,1
#     echo "[INFO] SLURM_JOB_GPUS not set, fallback to 0,1"
# else
#     export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
#     echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
# fi


# echo "[INFO] Starting vLLM (internvl) server on GPU 0..."
# source /home/wiss/zhang/anaconda3/bin/activate vllm

# CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# vllm serve openbmb/MiniCPM-V-2_6 \
#     --served-model-name minicpm \
#     --port 8000 \
#     --limit-mm-per-prompt image=20 \
#     --trust-remote-code &
# VLLM_PID=$!


# echo "[INFO] Waiting for vLLM (internvl) server to be ready..."
# for i in {1..300}; do
#     if curl -s http://localhost:8000/v1/models > /dev/null; then
#         echo "[INFO] ✅ internvl API is ready!"
#         break
#     fi
#     echo "  ... waiting ($((i*10))s)"
#     sleep 10
#     if [ $i -eq 300 ]; then
#         echo "[ERROR] ❌ Timeout: internvl server failed to start."
#         if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
#             kill "$VLLM_PID"
#         fi
#         exit 1
#     fi
# done


# echo "[INFO] Starting AEQA evaluation on GPU 1 (3dmem env)..."
source /home/wiss/zhang/anaconda3/bin/activate 3dmem
source .env
export LD_LIBRARY_PATH=/home/wiss/zhang/local_cuda118/cuda_cudart/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

python -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
 /home/wiss/zhang/code/openeqa/3D-Mem/run_aeqa_evaluation_gpt.py \
    -cf /home/wiss/zhang/code/openeqa/3D-Mem/cfg/eval_aeqa_debug.yaml


# echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
# if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
#     kill "$VLLM_PID"
# else
#     echo "[WARN] No running vLLM process to kill"
# fi

# echo "=== JOB END ==="
