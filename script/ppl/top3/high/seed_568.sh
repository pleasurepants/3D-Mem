#!/bin/bash
#SBATCH --job-name=llm-top3-high-568
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/llm_score/top3/high/seed_568-%j.out
#SBATCH --partition a40



# srun --nodes=1 --gres=gpu:a40:2 --ntasks=1 --cpus-per-task=16 --time=1:00:00 --partition a40 --pty bash
# srun --nodes=1 --gres=gpu:a40:2 --ntasks=1 --cpus-per-task=16 --time=0:30:00 --partition a40 --pty bash

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



vllm serve /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-VL-7B-Instruct \
    --served-model-name qwen \
    --port 8000 \
    --max-model-len 100000 \
    --limit-mm-per-prompt '{"image": 20}' &
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
# -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
CUDA_VISIBLE_DEVICES=1 python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/script/ppl/top3/high/config_568.yaml \
    --replay_mode traj_sim \
    --replay_top 3 \
    --retrieve_root /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set \
    --chat_seed 568 \
    --traj_file /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/traj_abs_format.json \
    --caption true \
    --critique true \
    --abstraction true \
    --exp_tuple /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/exp_tuple_v0.json \
    --ppl_rank high \
    --ppl_rank_file /home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/score/traj_abs_format_score_structured.json


echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="

