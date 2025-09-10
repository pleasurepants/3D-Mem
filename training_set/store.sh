#!/bin/bash
#SBATCH --job-name=get-embeddings
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/experience/get-embeddings-%j.out
#SBATCH --partition a40

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH
# srun --nodes=1 --gres=gpu:a100:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:2 --ntasks=1 --cpus-per-task=16 --time=1:00:00 --partition a40 --pty bash


unset http_proxy; unset https_proxy; unset HTTP_PROXY; unset HTTPS_PROXY

export CONDA_PREFIX=/home/atuin/v100dd/v100dd12/software/private/conda/envs/3dmem
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PILLOW_DISABLE_LERC=1
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false MAGNUM_LOG=quiet

python -m training_set.build_question_store \
  --questions_path /home/hpc/v100dd/v100dd12/code/3D-Mem/data/aeqa_questions-168.json \
  --dst_root /anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168/retrieve \
  --sbert_model /anvme/workspace/v100dd12-3dmem/model/all-MiniLM-L6-v2 \
  --faiss_factory_txt Flat