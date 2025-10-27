# PPL_RANK 功能使用说明

## 功能概述

`ppl_rank` 功能允许根据困惑度（perplexity）等级过滤检索池，只在特定类别的 question_id 中进行召回。

## 参数说明

### `--ppl_rank`
- **类型**: 字符串
- **可选值**: `low`, `medium`, `high` (不区分大小写)
- **默认值**: 空 (不启用过滤)
- **说明**: 指定要检索的困惑度类别

### `--ppl_rank_file`
- **类型**: 字符串
- **默认值**: `/home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/traj_abs_format_ppl_rank.json`
- **说明**: ppl_rank 配置文件的路径

## 使用方法

### 基本用法

在运行脚本时添加 `--ppl_rank` 参数：

```bash
python run_aeqa_evaluation_qwen.py \
    -cf cfg/eval_aeqa_debug.yaml \
    --replay_mode traj_random \
    --ppl_rank low \
    --其他参数...
```

### 不同类别的使用示例

#### Low 类别 (低困惑度, 55个问题)
```bash
--ppl_rank low
```

#### Medium 类别 (中等困惑度, 54个问题)
```bash
--ppl_rank medium
```

#### High 类别 (高困惑度, 54个问题)
```bash
--ppl_rank high
```

### 自定义 ppl_rank 文件
```bash
--ppl_rank low \
--ppl_rank_file /path/to/your/custom_ppl_rank.json
```

## 配置文件格式

ppl_rank 文件应遵循以下格式：

```json
{
  "summary": {
    "total_samples": 163,
    "rank_distribution": {
      "Low": {
        "count": 55,
        "mean_perplexity": 7.706394975835627
      },
      "Medium": {...},
      "High": {...}
    }
  },
  "Low": {
    "question_id_1": {
      "abstraction": "...",
      "perplexity": 7.638510704040527
    },
    ...
  },
  "Medium": {...},
  "High": {...}
}
```

## 工作原理

1. **初始化阶段**: 
   - 在 `FrontierSimilaritySearcher` 初始化时，加载 ppl_rank 配置文件
   - 根据指定的类别 (low/medium/high) 提取对应的 question_id 列表

2. **检索阶段**:
   - 在遍历候选 frontier 时，检查每个候选的 question_id
   - 如果 question_id 不在指定类别的列表中，则跳过该候选
   - 只保留属于指定类别的候选进行相似度计算

3. **日志输出**:
   - 启用时会输出: `[PPL_RANK] Mode enabled: category=low, file=...`
   - 加载成功会输出: `[PPL_RANK] Loaded X question_ids for category 'Low'`

## 数据统计

- **Low**: 55 个问题，平均困惑度 7.71
- **Medium**: 54 个问题，平均困惑度 9.54
- **High**: 54 个问题，平均困惑度 11.85

## 注意事项

1. 类别名称不区分大小写 (low/Low/LOW 都可以)
2. 如果不指定 `--ppl_rank`，过滤功能不会启用，保持原有行为
3. 如果配置文件不存在或格式错误，会自动禁用过滤并输出警告
4. 过滤仅影响检索池，不影响其他检索逻辑和参数

## 完整示例

### qwen_debug.sh 中的使用

```bash
CUDA_VISIBLE_DEVICES=1 python -m debugpy --listen 0.0.0.0:8798 --wait-for-client \
 /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py \
    -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/cfg/eval_aeqa_debug.yaml \
    --replay_mode traj_random \
    --replay_top 3 \
    --use_episodic_context 1 \
    --retrieve_root /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set \
    --traj_file /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/traj_abs_single.json \
    --exp_tuple /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/exp_tuple_v0.json \
    --exp_at cvf \
    --chat_seed 32 \
    --caption true \
    --critique true \
    --abstraction true \
    --ppl_rank low \
    --ppl_rank_file /home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/traj_abs_format_ppl_rank.json
```

## 修改的文件

1. **run_aeqa_evaluation_qwen.py**
   - 添加了 `--ppl_rank` 和 `--ppl_rank_file` 命令行参数
   - 将参数传递给 cfg 对象

2. **src/context_generator.py**
   - 在 `FrontierSimilaritySearcher.__init__` 中添加了 `_load_ppl_rank_filter()` 方法
   - 在 `_candidate_iter()` 中添加了过滤逻辑

3. **script/alex/qwen_debug.sh**
   - 添加了示例参数使用

## 测试结果

✅ 文件加载正常  
✅ 三个类别数据完整  
✅ 过滤逻辑正确工作  
✅ 无语法错误  

