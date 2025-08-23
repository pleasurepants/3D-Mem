#!/bin/bash

# 续接未完成任务的脚本
# 用于重新提交因为时间限制而停止的任务

echo "=== 续接未完成任务脚本 ==="
date

# 检查每个seed的任务状态
for seed in 100 200 300; do
    echo -e "\n检查 Seed ${seed} 的任务状态..."
    
    # 检查输出目录
    output_dir="/home/atuin/v100dd/v100dd12/openeqa/env_epi_v0/qwen/seed${seed}"
    
    # 检查是否有结果文件
    if [ -d "$output_dir" ]; then
        result_files=$(find "$output_dir" -name "*.json" -type f 2>/dev/null | wc -l)
        echo "  结果文件数量: $result_files"
        
        # 如果结果文件数量少于预期，认为任务未完成
        if [ "$result_files" -lt 10 ]; then  # 假设应该有至少10个结果文件
            echo "  Seed ${seed} 任务可能未完成，重新提交..."
            
            # 重新提交任务
            case $seed in
                100)
                    echo "  重新提交 seed100 任务..."
                    sbatch script/alex/alex_qwen_seed100.sh
                    ;;
                200)
                    echo "  重新提交 seed200 任务..."
                    sbatch script/alex/alex_qwen_seed200.sh
                    ;;
                300)
                    echo "  重新提交 seed300 任务..."
                    sbatch script/alex/alex_qwen_seed300.sh
                    ;;
            esac
        else
            echo "  Seed ${seed} 任务已完成，无需重新提交"
        fi
    else
        echo "  Seed ${seed} 输出目录不存在，重新提交任务..."
        
        # 重新提交任务
        case $seed in
            100)
                echo "  重新提交 seed100 任务..."
                sbatch script/alex/alex_qwen_seed100.sh
                ;;
            200)
                echo "  重新提交 seed200 任务..."
                sbatch script/alex/alex_qwen_seed200.sh
                ;;
            300)
                echo "  重新提交 seed300 任务..."
                sbatch script/alex/alex_qwen_seed300.sh
                ;;
        esac
    fi
done

echo -e "\n=== 续接任务完成 ==="
echo "当前运行的任务："
squeue -u $USER
