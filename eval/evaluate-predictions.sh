#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/evaliuat.out 
#SBATCH --partition a40

source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem

# srun --nodes=1 --gres=gpu:a100:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a100 --pty bash
# srun --nodes=1 --gres=gpu:a40:2 --ntasks=1 --cpus-per-task=16 --time=4:00:00 --partition a40 --pty bash

date
hostname
which python

export NCCL_P2P_DISABLE=1
export PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/openeqa/bin:$PATH
MASTER_ADDR=localhost

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

echo $PYTHONPATH

dir_path="/anvme/workspace/v100dd12-3dmem/output/qwen/hierarchical/hiera-only-vote"


question_num=41
echo "Running on MASTER_NODE=$MASTER_NODE, MASTER_PORT=$MASTER_PORT, RDZV_ID=$RDZV_ID"

python /home/hpc/v100dd/v100dd12/code/3D-Mem/eval/evaluate-predictions.py \
    --dataset /home/hpc/v100dd/v100dd12/code/3D-Mem/eval/open-eqa-41.json \
    --output-directory ${dir_path} \
    --results ${dir_path}/gpt_answer.json
    
python /home/hpc/v100dd/v100dd12/code/3D-Mem/eval/get-scores_csv.py \
    --dataset open-eqa-${question_num} \
    --result-path ${dir_path} \

    