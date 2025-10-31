#!/bin/bash
#SBATCH --job-name=goatbench_build_retrieve_store
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
# export END_POINT="http://10.153.51.154:8006/v1"    # worker-5



# python cursor/build_frontier_ahash_index.py \
#   --root /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13 \
#   --workers 16 \
#   --hash_size 8 \
#   --merge

# [Index] ========== Summary ==========
# [Index] root=/nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13
# [Index] images scanned=125381
# [Index] updated=125381, skipped=0, failed=0
# [Index] total entries in index=125381
# [Index] output=/nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/.frontier_ahash_index.json
# [Index] total time=2074.06s (34.57 minutes)
# [Index] average speed=60.45 files/s
# [Index] ==============================



python -m training_set.build_retrieve_store \
  --questions_path /nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/question_list.json \
  --dst_root /anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168/retrieve \
  --sbert_model /anvme/workspace/v100dd12-3dmem/model/all-MiniLM-L6-v2 \
  --faiss_factory_txt Flat