#!/bin/bash
#SBATCH --job-name=goatbench_hierarchical-cot_seed19
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --nodelist=worker-3
#SBATCH --output=/nfs/data8/jingpei/eqa/3D-Mem/slurm/goatbench/%x-%j.out 


source /home/wiss/jingpei/anaconda3/bin/activate 
conda activate 3dmem

date
hostname
which python



cd /nfs/data8/jingpei/eqa/3D-Mem

export END_POINT="http://10.153.51.155:8008/v1"


# python run_goatbench_evaluation.py -cf cfg/dbs_cfg/goatbench_qwen_13.yaml
python run_goatbench_evaluation_qwen.py -cf cfg/dbs_cfg/goatbench_qwen_13.yaml --replay_top 0