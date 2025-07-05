#!/bin/bash
#SBATCH --job-name=debug_mini_test
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/video-blip-icl/slurm_storage/bash_test-%j.out 
#SBATCH --partition a40

# echo "=== JOB START ==="
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source .env
export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

python -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
 /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_internvl.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/cfg/alex/eval_aeqa_debug.yaml

