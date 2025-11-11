#!/bin/bash
#SBATCH --job-name=ug-s-t5-s32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=16:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/experience/format_unformat/unformat/gpt_qwenabs/sim/top-5/unformat-all-sim-t5-s32-%j.out
#SBATCH --partition a40

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80
echo "=== JOB START ==="
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

echo "[INFO] Starting AEQA evaluation (gpt, OpenAI API)..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source .env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_gpt.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/script/experience/format_unformat/unformat/gpt/sim/top-5/unformat-all-sim-t5-s32.yaml \
    --replay_mode traj_sim \
    --replay_top 5 \
    --retrieve_root /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set \
    --chat_seed 32 \
    --traj_file /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/experience/unformat.json \
    --exp_tuple /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/exp_tuple_v0.json

echo "=== JOB END ==="
