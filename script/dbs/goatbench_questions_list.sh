#!/bin/bash
#SBATCH --job-name=goatbench_questions_list_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=worker-8
#SBATCH --output=/nfs/data8/jingpei/eqa/3D-Mem/slurm/goatbench/%x-%j.out 


source /home/wiss/jingpei/anaconda3/bin/activate 
conda activate 3dmem

date
hostname
which python



cd /nfs/data8/jingpei/eqa/3D-Mem

# export END_POINT="http://10.153.51.155:8009/v1"    # worker-6
# export END_POINT="http://10.153.51.154:8009/v1"    # worker-5
# export END_POINT="http://10.153.51.154:8006/v1"    # worker-5

python generate_questions_list.py -cf /nfs/data8/jingpei/eqa/3D-Mem/cfg/dbs_cfg/goatbench_qwen_rollout.yaml
# in train_hierarchical-cot_log
# python generate_questions_list.py -cf /nfs/data8/jingpei/eqa/3D-Mem/cfg/dbs_cfg/goatbench_qwen_seed.yaml
# # in hierarchical-cot_log