#!/bin/bash

# 综合检查和续接脚本
# 用于定期检查任务状态并在需要时重新提交

echo "=== 任务状态检查和续接 ==="
date

# 检查当前运行的任务
echo "当前运行的任务："
squeue -u $USER

# 检查是否有任务因为时间限制而停止
echo -e "\n检查最近停止的任务："
recent_jobs=$(sacct -u $USER --starttime=$(date -d '2 hours ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,ExitCode,Elapsed,TimeLimit | grep -E "(TIMEOUT|CANCELLED|FAILED)" | head -10)

if [ -n "$recent_jobs" ]; then
    echo "发现停止的任务："
    echo "$recent_jobs"
    
    # 检查每个seed的任务完成情况
    for seed in 100 200 300; do
        echo -e "\n检查 Seed ${seed} 的完成情况..."
        
        output_dir="/home/atuin/v100dd/v100dd12/openeqa/env_epi_v0/qwen/seed${seed}"
        
        if [ -d "$output_dir" ]; then
            # 计算结果文件数量
            result_files=$(find "$output_dir" -name "*.json" -type f 2>/dev/null | wc -l)
            echo "  结果文件数量: $result_files"
            
            # 检查是否有最终结果文件
            final_result=$(find "$output_dir" -name "*final_results*.json" -type f 2>/dev/null | head -1)
            
            if [ -n "$final_result" ]; then
                echo "  找到最终结果文件: $final_result"
                echo "  Seed ${seed} 任务已完成"
            else
                echo "  未找到最终结果文件，任务可能未完成"
                
                # 检查是否有正在运行的任务
                running_job=$(squeue -u $USER | grep "q-envepi_v0-${seed}" | wc -l)
                if [ "$running_job" -eq 0 ]; then
                    echo "  没有运行中的任务，重新提交..."
                    
                    case $seed in
                        100)
                            sbatch script/alex/alex_qwen_seed100.sh
                            echo "  已重新提交 seed100 任务"
                            ;;
                        200)
                            sbatch script/alex/alex_qwen_seed200.sh
                            echo "  已重新提交 seed200 任务"
                            ;;
                        300)
                            sbatch script/alex/alex_qwen_seed300.sh
                            echo "  已重新提交 seed300 任务"
                            ;;
                    esac
                else
                    echo "  任务正在运行中"
                fi
            fi
        else
            echo "  输出目录不存在，重新提交任务..."
            
            # 检查是否有正在运行的任务
            running_job=$(squeue -u $USER | grep "q-envepi_v0-${seed}" | wc -l)
            if [ "$running_job" -eq 0 ]; then
                case $seed in
                    100)
                        sbatch script/alex/alex_qwen_seed100.sh
                        echo "  已重新提交 seed100 任务"
                        ;;
                    200)
                        sbatch script/alex/alex_qwen_seed200.sh
                        echo "  已重新提交 seed200 任务"
                        ;;
                    300)
                        sbatch script/alex/alex_qwen_seed300.sh
                        echo "  已重新提交 seed300 任务"
                        ;;
                esac
            else
                echo "  任务正在运行中"
            fi
        fi
    done
else
    echo "没有发现最近停止的任务"
fi

echo -e "\n=== 检查完成 ==="
echo "当前所有任务状态："
squeue -u $USER
