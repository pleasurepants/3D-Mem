#!/bin/bash
#SBATCH --job-name=goatbench_generate_experience_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=worker-minor-6
#SBATCH --output=/nfs/data8/jingpei/eqa/3D-Mem/slurm/goatbench/%x-%j.out 


source /home/wiss/jingpei/anaconda3/bin/activate 
conda activate 3dmem

date
hostname
which python



cd /nfs/data8/jingpei/eqa/3D-Mem

# export END_POINT="http://10.153.51.155:8009/v1"    # worker-6
# export END_POINT="http://10.153.51.154:8009/v1"    # worker-5
export END_POINT="http://10.153.51.154:8006/v1"    # worker-5

# generate caption and experience
# python generate_experience_from_json_goatbench.py \
#     --input_json /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/replay_step_info.json \
#     --output_json /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/experience_output.json \
#     --output_parent_dir /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/ \
#     --exp_name seed13

# parse experience into critique and abstraction, rearrange structure
python generate_experience_from_json_goatbench.py \
    --input_json /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/replay_step_info.json \
    --output_json /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/experience_output.json \
    --output_parent_dir /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/ \
    --exp_name seed13 \
    --captions_only \
    --experience_json_path /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/experience_output.json \

# python generate_abstraction_from_captions.py \
#   --exp_tuple /anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168/exp_tuple_v0.json \
#   --max_steps 10 \
#   --seed 32 \
#   --out /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/traj_abs_single.json