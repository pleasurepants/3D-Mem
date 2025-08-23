#!/bin/bash

# 24小时超时检查脚本
# 专门用于检查任务是否因为24小时时间限制而停止

CURSOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$CURSOR_DIR")"

echo "=== 24小时超时检查 ==="
date
echo ""

cd "$PROJECT_ROOT"

# 检查最近24小时内停止的任务
echo "检查最近24小时内停止的任务："
timeout_jobs=$(sacct -u $USER --starttime=$(date -d '24 hours ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,ExitCode,Elapsed,TimeLimit | grep -E "(TIMEOUT|CANCELLED|FAILED)")

if [ -n "$timeout_jobs" ]; then
    echo "发现超时或停止的任务："
    echo "$timeout_jobs"
    echo ""
    
    # 检查每个seed的任务完成情况
    for seed in 100 200 300; do
        echo "检查 Seed ${seed} 的完成情况..."
        
        output_dir="/home/atuin/v100dd/v100dd12/openeqa/env_epi_v0/qwen/seed${seed}"
        
        # 检查是否有最终结果文件
        final_result=$(find "$output_dir" -name "*final_results*.json" -type f 2>/dev/null | head -1)
        
        if [ -n "$final_result" ]; then
            echo "  ✅ Seed ${seed} 任务已完成"
        else
            echo "  ❌ Seed ${seed} 任务未完成"
            
            # 检查是否有正在运行的任务
            running_job=$(squeue -u $USER | grep "q-envepi_v0-${seed}" | wc -l)
            
            if [ "$running_job" -eq 0 ]; then
                echo "  🔄 重新提交 Seed ${seed} 任务..."
                
                case $seed in
                    100)
                        sbatch script/alex/alex_qwen_seed100.sh
                        echo "    已重新提交 seed100 任务"
                        ;;
                    200)
                        sbatch script/alex/alex_qwen_seed200.sh
                        echo "    已重新提交 seed200 任务"
                        ;;
                    300)
                        sbatch script/alex/alex_qwen_seed300.sh
                        echo "    已重新提交 seed300 任务"
                        ;;
                esac
            else
                echo "  ⏳ Seed ${seed} 任务正在运行中"
            fi
        fi
        echo ""
    done
else
    echo "没有发现最近24小时内超时或停止的任务"
fi

echo "=== 检查完成 ==="
echo "当前任务状态："
squeue -u $USER
