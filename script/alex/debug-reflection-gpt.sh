#!/bin/bash
#SBATCH --job-name=unformat-gpt
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=8:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/experience/unformat-gpt-traj_version-%j.out
#SBATCH --partition a40



# srun --nodes=1 --gres=gpu:a100:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a40 --pty bash


export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH



source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source /home/hpc/v100dd/v100dd12/code/3D-Mem/.env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
python /home/hpc/v100dd/v100dd12/code/3D-Mem/experience_gpt.py \
  --chunk_caption /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/final_captions_from_chunck.json \
  --seed 32 \
  --exp_mode unformat \
  --out /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/experience/unformat-gpt.json




echo "=== JOB END ==="
