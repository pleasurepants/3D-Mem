#!/bin/bash
#SBATCH --job-name=goatbench_experience_full_unformat_top5_seed0
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=10-00:00:00
#SBATCH --nodelist=worker-9
#SBATCH --output=/nfs/data8/jingpei/eqa/3D-Mem/slurm/goatbench/%x-%j.out 


source /home/wiss/jingpei/anaconda3/bin/activate 
conda activate 3dmem

date
hostname
which python



cd /nfs/data8/jingpei/eqa/3D-Mem

# export END_POINT="http://10.153.51.155:8009/v1"    # worker-6
# export END_POINT="http://10.153.51.154:8009/v1"    # worker-5
export END_POINT="http://10.153.51.154:8009/v1"    # worker-5


# # python run_goatbench_evaluation.py -cf cfg/dbs_cfg/goatbench_qwen_13.yaml
# python run_goatbench_evaluation_qwen.py \
#     -cf cfg/dbs_cfg/goatbench_qwen_experience.yaml \
#     --replay_mode traj_sim \
#     --replay_top 3 \
#     --retrieve_root /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/ \
#     --traj_file /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/captions_reflection_abstraction.json \
#     --caption true \
#     --critique true \
#     --abstraction true \
#     --exp_tuple /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/exp_tuple_v0.json \
#     # --chat_seed 13

# unformat
python run_goatbench_evaluation_qwen.py \
    -cf cfg/dbs_cfg/goatbench_qwen_experience.yaml \
    --replay_mode traj_sim \
    --replay_top 5 \
    --retrieve_root /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/ \
    --traj_file /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/captions_reflection_abstraction_unformat.json \
    --caption true \
    --critique true \
    --abstraction true \
    --exp_tuple /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/exp_tuple_v0.json \
    # --chat_seed 13