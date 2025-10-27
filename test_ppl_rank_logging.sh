#!/bin/bash

#SBATCH --job-name=test-ppl-log
#SBATCH --output=slurm/test_ppl_logging-%j.out
#SBATCH --error=slurm/test_ppl_logging-%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# 测试 ppl_rank 模式下的日志输出
cd /home/hpc/v100dd/v100dd12/code/3D-Mem

# 激活环境
source /home/atuin/v100dd/v100dd12/software/private/conda/bin/activate 3dmem

# 运行测试 - 使用一个简单的配置
python run_aeqa_evaluation_qwen.py \
    -cf script/ppl/top3/low/config_13.yaml \
    --ppl_rank low \
    --ppl_rank_file /home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/score/traj_abs_format_score_structured.json \
    --start_ratio 0.0 \
    --end_ratio 0.1 \
    --chat_seed 13
