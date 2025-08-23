#!/bin/bash

# 综合任务管理脚本
# 用于管理所有3D-Mem相关的任务

CURSOR_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$CURSOR_DIR")"

echo "=== 3D-Mem 任务管理器 ==="
echo "Cursor目录: $CURSOR_DIR"
echo "项目根目录: $PROJECT_ROOT"
echo "当前时间: $(date)"
echo ""

# 函数：显示帮助信息
show_help() {
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  status     - 显示当前任务状态"
    echo "  monitor    - 监控任务状态"
    echo "  resume     - 续接未完成的任务"
    echo "  submit     - 提交新任务"
    echo "  logs       - 查看日志"
    echo "  clean      - 清理旧日志"
    echo "  help       - 显示此帮助信息"
    echo ""
}

# 函数：显示任务状态
show_status() {
    echo "=== 当前任务状态 ==="
    squeue -u $USER
    
    echo -e "\n=== 最近完成的任务 ==="
    sacct -u $USER --starttime=$(date -d '1 hour ago' +%Y-%m-%dT%H:%M:%S) --format=JobID,JobName,State,ExitCode,Elapsed,TimeLimit
}

# 函数：监控任务
monitor_tasks() {
    echo "=== 任务监控 ==="
    cd "$PROJECT_ROOT"
    ./cursor/monitor_and_resume_tasks.sh
}

# 函数：续接任务
resume_tasks() {
    echo "=== 续接未完成任务 ==="
    cd "$PROJECT_ROOT"
    ./cursor/resume_incomplete_tasks.sh
}

# 函数：提交新任务
submit_tasks() {
    echo "=== 提交新任务 ==="
    cd "$PROJECT_ROOT"
    
    echo "可用的任务类型:"
    echo "1. qwen_seed100"
    echo "2. qwen_seed200" 
    echo "3. qwen_seed300"
    echo "4. 所有qwen任务"
    echo ""
    
    read -p "请选择任务类型 (1-4): " choice
    
    case $choice in
        1)
            echo "提交 qwen_seed100 任务..."
            sbatch script/alex/alex_qwen_seed100.sh
            ;;
        2)
            echo "提交 qwen_seed200 任务..."
            sbatch script/alex/alex_qwen_seed200.sh
            ;;
        3)
            echo "提交 qwen_seed300 任务..."
            sbatch script/alex/alex_qwen_seed300.sh
            ;;
        4)
            echo "提交所有 qwen 任务..."
            sbatch script/alex/alex_qwen_seed100.sh
            sbatch script/alex/alex_qwen_seed200.sh
            sbatch script/alex/alex_qwen_seed300.sh
            ;;
        *)
            echo "无效选择"
            return 1
            ;;
    esac
    
    echo "任务提交完成"
    squeue -u $USER
}

# 函数：查看日志
view_logs() {
    echo "=== 查看日志 ==="
    
    echo "选择要查看的日志类型:"
    echo "1. 当前运行任务的日志"
    echo "2. 特定seed的日志"
    echo "3. 最近的错误日志"
    echo ""
    
    read -p "请选择 (1-3): " log_choice
    
    case $log_choice in
        1)
            echo "当前运行任务的日志:"
            for jobid in $(squeue -u $USER -h -o "%i"); do
                echo "任务 $jobid 的日志:"
                scontrol show job $jobid | grep -E "(StdOut|StdErr)"
                echo ""
            done
            ;;
        2)
            read -p "请输入seed (100/200/300): " seed
            log_pattern="/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/env_epi_v0/qwen/seed${seed}-*.out"
            latest_log=$(ls -t $log_pattern 2>/dev/null | head -1)
            if [ -n "$latest_log" ]; then
                echo "Seed ${seed} 最新日志: $latest_log"
                echo "最后20行:"
                tail -20 "$latest_log"
            else
                echo "没有找到 seed ${seed} 的日志文件"
            fi
            ;;
        3)
            echo "最近的错误日志:"
            find /home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/env_epi_v0/qwen/ -name "*.out" -type f -exec grep -l "ERROR\|FAILED\|TIMEOUT" {} \; | head -5 | while read log; do
                echo "错误日志: $log"
                tail -10 "$log"
                echo ""
            done
            ;;
        *)
            echo "无效选择"
            ;;
    esac
}

# 函数：清理旧日志
clean_logs() {
    echo "=== 清理旧日志 ==="
    
    echo "警告：这将删除7天前的日志文件"
    read -p "确认继续？(y/N): " confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        find /home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/env_epi_v0/qwen/ -name "*.out" -type f -mtime +7 -delete
        echo "旧日志清理完成"
    else
        echo "取消清理"
    fi
}

# 主程序
case "${1:-help}" in
    status)
        show_status
        ;;
    monitor)
        monitor_tasks
        ;;
    resume)
        resume_tasks
        ;;
    submit)
        submit_tasks
        ;;
    logs)
        view_logs
        ;;
    clean)
        clean_logs
        ;;
    help|*)
        show_help
        ;;
esac
