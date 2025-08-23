#!/bin/bash

# 快速访问cursor文件夹中的任务管理工具
# 使用方法: ./cursor.sh [选项]

CURSOR_DIR="$(dirname "$0")/cursor"

if [ ! -d "$CURSOR_DIR" ]; then
    echo "错误：找不到cursor文件夹"
    exit 1
fi

# 如果没有参数，显示帮助信息
if [ $# -eq 0 ]; then
    echo "=== 3D-Mem 快速任务管理 ==="
    echo "使用方法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  status     - 显示当前任务状态"
    echo "  monitor    - 监控任务状态"
    echo "  resume     - 续接未完成的任务"
    echo "  submit     - 提交新任务"
    echo "  logs       - 查看日志"
    echo "  clean      - 清理旧日志"
    echo "  timeout    - 24小时超时检查"
    echo "  help       - 显示详细帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 status    # 查看任务状态"
    echo "  $0 timeout   # 检查24小时超时"
    echo "  $0 submit    # 提交新任务"
    exit 0
fi

# 根据参数调用相应的脚本
case "$1" in
    status|monitor|resume|submit|logs|clean)
        "$CURSOR_DIR/task_manager.sh" "$1"
        ;;
    timeout)
        "$CURSOR_DIR/check_24h_timeout.sh"
        ;;
    help)
        cat "$CURSOR_DIR/README.md"
        ;;
    *)
        echo "未知选项: $1"
        echo "使用 '$0 help' 查看帮助信息"
        exit 1
        ;;
esac
