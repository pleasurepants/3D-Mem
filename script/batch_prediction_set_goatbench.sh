#!/bin/bash
#SBATCH --job-name=41-goatbench
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-3,worker-4,worker-7
#SBATCH --output=/home/wiss/zhang/code/openeqa/3D-Mem/slurm/goatbench/41-goatbench-%j.out 
#SBATCH --partition all

# srun --pty --nodes=1 --ntasks=1 --cpus-per-task=16 --gres=gpu:1 --time=4:00:00 --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6 --partition all bash


source /home/wiss/zhang/anaconda3/bin/activate 3dmem
source .env

date
hostname
which python

export NCCL_P2P_DISABLE=1
export PATH=/home/wiss/zhang/anaconda3/envs/3dmem/bin:$PATH

MASTER_ADDR=localhost

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)




echo "Running on MASTER_NODE=$MASTER_NODE, MASTER_PORT=$MASTER_PORT, RDZV_ID=$RDZV_ID"

python -c "import omegaconf; print(omegaconf.__version__)"

python /home/wiss/zhang/code/openeqa/3D-Mem/run_goatbench_evaluation.py -cf /home/wiss/zhang/code/openeqa/3D-Mem/cfg/eval_goatbench.yaml