#!/bin/bash
#SBATCH --job-name=tar
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=23:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/tar-%j.out
#SBATCH --partition a40

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH


tar -czf /anvme/workspace/v100dd12-3dmem_rebuttal/3dmem_workspace.tar.gz /anvme/workspace/v100dd12-3dmem_rebuttal/v100dd12-3mem-1769055061/
