#!/bin/bash
#SBATCH --job-name=gpt-baseline
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/gpt/baseline-%j.out 
#SBATCH --partition a100



# srun --nodes=1 --gres=gpu:a100:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a40 --pty bash


export http_proxy=http://proxy.nhr.fau.de:80
export https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=http://proxy.nhr.fau.de:80
export HTTPS_PROXY=http://proxy.nhr.fau.de:80

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH



source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source /home/hpc/v100dd/v100dd12/code/3D-Mem/.env


echo "[INFO] Starting AEQA evaluation (3dmem env)..."
# export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/cfg/alex_cfg/gpt.yaml


echo "=== JOB END ==="
