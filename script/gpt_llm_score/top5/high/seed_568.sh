#!/bin/bash
#SBATCH --job-name=llm-top5-high-568
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/gpt/llm_score/top5-high-seed_568-%j.out
#SBATCH --partition a40



# srun --nodes=1 --gres=gpu:a40:1 --ntasks=1 --cpus-per-task=16 --time=1:00:00 --partition a40 --pty bash
# srun --nodes=1 --gres=gpu:a40:1 --ntasks=1 --cpus-per-task=16 --time=0:30:00 --partition a40 --pty bash

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH



source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source /home/hpc/v100dd/v100dd12/code/3D-Mem/.env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_gpt.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/script/ppl/top5/high/config_568.yaml \
    --replay_mode traj_sim \
    --replay_top 5 \
    --retrieve_root /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set \
    --chat_seed 568 \
    --traj_file /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/experience/unformat-gpt.json \
    --exp_tuple /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/exp_tuple_v0.json \
    --ppl_rank high \
    --ppl_rank_file /home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/score/gpt/gpt_score_structured.json


echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="

