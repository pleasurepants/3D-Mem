#!/bin/bash
#SBATCH --job-name=goatbench
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-3,worker-4,worker-6
#SBATCH --output=/home/wiss/zhang/code/openeqa/3D-Mem/slurm/run-goatbench-%j.out 
#SBATCH --partition all

source /home/wiss/zhang/anaconda3/bin/activate 3dmem

date
hostname
which python

export NCCL_P2P_DISABLE=1
export PATH=/home/wiss/zhang/anaconda3/envs/3dmem/bin:$PATH

MASTER_ADDR=localhost

RDZV_ID=$RANDOM
MASTER_NODE=$(hostname)
MASTER_PORT=$(comm -23 <(seq 8000 9000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

echo $PYTHONPATH



echo "Running on MASTER_NODE=$MASTER_NODE, MASTER_PORT=$MASTER_PORT, RDZV_ID=$RDZV_ID"

python -c "import omegaconf; print(omegaconf.__version__)"

python run_goatbench_evaluation.py -cf cfg/eval_goatbench.yaml