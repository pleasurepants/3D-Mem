#!/bin/bash

# 监控和续接任务的脚本
# 用于检查任务是否因为时间限制而停止，并重新提交

echo "=== 任务监控和续接脚本 ==="
date

# 检查当前运行的任务
echo "当前运行的任务："
squeue -u $USER

# 检查最近完成的任务
echo -e "\n最近完成的任务："
sacct -u $USER --starttime=$(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,ExitCode,Elapsed,TimeLimit

# 检查是否有任务因为时间限制而停止
echo -e "\n检查是否有任务因为时间限制而停止："
sacct -u $USER --starttime=$(date -d '2 hours ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,ExitCode,Elapsed,TimeLimit | grep -E "(TIMEOUT|CANCELLED)"

# 检查输出目录，判断任务是否完成
echo -e "\n检查任务输出目录："
for seed in 100 200 300; do
    output_dir="/home/atuin/v100dd/v100dd12/openeqa/env_epi_v0/qwen/seed${seed}"
    if [ -d "$output_dir" ]; then
        echo "Seed ${seed} 输出目录存在: $output_dir"
        # 检查是否有结果文件
        result_files=$(find "$output_dir" -name "*.json" -type f 2>/dev/null | wc -l)
        echo "  结果文件数量: $result_files"
    else
        echo "Seed ${seed} 输出目录不存在: $output_dir"
    fi
done

# 检查日志文件
echo -e "\n检查最近的日志文件："
for seed in 100 200 300; do
    log_pattern="/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/env_epi_v0/qwen/seed${seed}-*.out"
    latest_log=$(ls -t $log_pattern 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "Seed ${seed} 最新日志: $latest_log"
        # 检查日志最后几行
        echo "  最后几行日志："
        tail -5 "$latest_log" 2>/dev/null | sed 's/^/    /'
    else
        echo "Seed ${seed} 没有找到日志文件"
    fi
done

echo -e "\n=== 监控完成 ==="
