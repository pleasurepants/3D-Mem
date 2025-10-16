# PPL Rank 实验脚本

这个目录包含了按照 PPL (Perplexity) Rank 和 Replay Top-K 分类的实验脚本。

## 目录结构

```
script/ppl/
├── top1/         # replay_top = 1
│   ├── low/
│   │   ├── seed_13.sh, seed_32.sh, seed_568.sh
│   │   └── config_13.yaml, config_32.yaml, config_568.yaml
│   ├── medium/
│   │   └── (同上)
│   └── high/
│       └── (同上)
├── top3/         # replay_top = 3
│   ├── low/
│   ├── medium/
│   └── high/
└── top5/         # replay_top = 5
    ├── low/
    ├── medium/
    └── high/
```

**总计：27个shell脚本 + 27个yaml配置文件 = 54个文件**

## 配置说明

### 固定参数
- `--replay_mode`: traj_sim
- 所有其他参数与原模板保持一致

### 变化参数

| Top-K | PPL Rank | Seeds | --replay_top | 输出路径示例 |
|-------|----------|-------|--------------|-------------|
| top1 | low | 13, 32, 568 | 1 | /anvme/.../trajectory_exp/ppl/top1/low/seed_13/ |
| top1 | medium | 13, 32, 568 | 1 | /anvme/.../trajectory_exp/ppl/top1/medium/seed_13/ |
| top1 | high | 13, 32, 568 | 1 | /anvme/.../trajectory_exp/ppl/top1/high/seed_13/ |
| top3 | low | 13, 32, 568 | 3 | /anvme/.../trajectory_exp/ppl/top3/low/seed_13/ |
| top3 | medium | 13, 32, 568 | 3 | /anvme/.../trajectory_exp/ppl/top3/medium/seed_13/ |
| top3 | high | 13, 32, 568 | 3 | /anvme/.../trajectory_exp/ppl/top3/high/seed_13/ |
| top5 | low | 13, 32, 568 | 5 | /anvme/.../trajectory_exp/ppl/top5/low/seed_13/ |
| top5 | medium | 13, 32, 568 | 5 | /anvme/.../trajectory_exp/ppl/top5/medium/seed_13/ |
| top5 | high | 13, 32, 568 | 5 | /anvme/.../trajectory_exp/ppl/top5/high/seed_13/ |

## 使用方法

### 提交单个任务
```bash
cd /home/hpc/v100dd/v100dd12/code/3D-Mem
sbatch script/ppl/top3/low/seed_13.sh
```

### 批量提交特定top-k的所有实验
```bash
cd /home/hpc/v100dd/v100dd12/code/3D-Mem
# 提交所有top3实验 (9个任务)
for ppl in low medium high; do
    for seed in 13 32 568; do
        sbatch script/ppl/top3/${ppl}/seed_${seed}.sh
    done
done
```

### 批量提交特定ppl级别的所有top-k实验
```bash
cd /home/hpc/v100dd/v100dd12/code/3D-Mem
# 提交所有low ppl实验 (9个任务)
for top in top1 top3 top5; do
    for seed in 13 32 568; do
        sbatch script/ppl/${top}/low/seed_${seed}.sh
    done
done
```

### 批量提交全部27个任务
```bash
cd /home/hpc/v100dd/v100dd12/code/3D-Mem
for top in top1 top3 top5; do
    for ppl in low medium high; do
        for seed in 13 32 568; do
            sbatch script/ppl/${top}/${ppl}/seed_${seed}.sh
        done
    done
done
```

## 输出位置

### 日志文件
- Top1: `/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/ppl/top1/{low,medium,high}/seed_{13,32,568}-*.out`
- Top3: `/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/ppl/top3/{low,medium,high}/seed_{13,32,568}-*.out`
- Top5: `/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/ppl/top5/{low,medium,high}/seed_{13,32,568}-*.out`

### 结果文件
所有实验结果输出到：
- `/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/trajectory_exp/ppl/{top1,top3,top5}/{low,medium,high}/seed_{13,32,568}/`

## PPL Rank 说明

- **Low**: 55个问题，平均困惑度 7.71，范围 5.56-8.87
- **Medium**: 54个问题，平均困惑度 9.54，范围 8.87-10.21
- **High**: 54个问题，平均困惑度 11.85，范围 10.25-17.80

## Replay Top-K 说明

- **top1**: 检索Top-1最相似的trajectory abstraction
- **top3**: 检索Top-3最相似的trajectory abstraction
- **top5**: 检索Top-5最相似的trajectory abstraction

## 注意事项

1. 所有脚本都已添加可执行权限
2. 每个脚本使用2个A40 GPU（GPU0运行vLLM，GPU1运行评估）
3. 最大运行时间设置为12小时
4. 确保在正确的conda环境下运行（vllm和3dmem）

## 实验组合矩阵

| Top-K | PPL Rank | Seeds | 任务数 |
|-------|----------|-------|-------|
| 3种 | 3种 | 3个 | **27个** |

每个组合都有独立的输出路径和job名称，便于区分和管理。
