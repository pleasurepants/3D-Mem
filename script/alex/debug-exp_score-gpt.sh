#!/bin/bash
#SBATCH --job-name=exp-score
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=5:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/experience/exp-score-gpt-%j.out
#SBATCH --partition a40



# srun --nodes=1 --gres=gpu:a100:1 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:1 --ntasks=1 --cpus-per-task=16 --time=1:00:00 --partition a40 --pty bash

export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

echo "[INFO] Starting AEQA evaluation ..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source /home/hpc/v100dd/v100dd12/code/3D-Mem/.env
# export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
CUDA_VISIBLE_DEVICES=0 python /home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/score/gpt_score.py \
  --input_json /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/experience/unformat-gpt.json \
  --output_json /home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/score/unformat-gpt-llm_score.json




echo "=== JOB END ==="
