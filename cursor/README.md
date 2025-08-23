# Cursor 任务管理工具

这个文件夹包含了所有用于管理3D-Mem项目的脚本和工具。

## 脚本说明

### 主要脚本

1. **task_manager.sh** - 综合任务管理器
   - 使用方法: `./task_manager.sh [选项]`
   - 选项:
     - `status` - 显示当前任务状态
     - `monitor` - 监控任务状态
     - `resume` - 续接未完成的任务
     - `submit` - 提交新任务
     - `logs` - 查看日志
     - `clean` - 清理旧日志
     - `help` - 显示帮助信息

2. **check_24h_timeout.sh** - 24小时超时检查
   - 专门用于检查任务是否因为24小时时间限制而停止
   - 自动重新提交未完成的任务

3. **monitor_and_resume_tasks.sh** - 任务监控脚本
   - 监控当前运行的任务
   - 检查输出目录和日志文件

4. **resume_incomplete_tasks.sh** - 续接未完成任务
   - 检查任务完成情况
   - 重新提交未完成的任务

5. **check_and_resume.sh** - 综合检查和续接
   - 结合监控和续接功能

## 使用方法

### 快速检查任务状态
```bash
./task_manager.sh status
```

### 监控任务
```bash
./task_manager.sh monitor
```

### 24小时超时检查
```bash
./check_24h_timeout.sh
```

### 提交新任务
```bash
./task_manager.sh submit
```

### 查看日志
```bash
./task_manager.sh logs
```

## 任务类型

当前支持的任务类型：
- qwen_seed100 (seed=100)
- qwen_seed200 (seed=200)
- qwen_seed300 (seed=300)

## 输出目录

任务结果保存在：
- `/home/atuin/v100dd/v100dd12/openeqa/env_epi_v0/qwen/seed100/`
- `/home/atuin/v100dd/v100dd12/openeqa/env_epi_v0/qwen/seed200/`
- `/home/atuin/v100dd/v100dd12/openeqa/env_epi_v0/qwen/seed300/`

## 日志文件

日志文件保存在：
- `/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/env_epi_v0/qwen/`

## 注意事项

1. 所有脚本都会自动检查任务完成情况
2. 如果任务因为时间限制而停止，会自动重新提交
3. 脚本会检查最终结果文件来判断任务是否真正完成
4. 建议定期运行 `check_24h_timeout.sh` 来检查超时任务

## 定时任务建议

可以设置cron job来定期检查：
```bash
# 每小时检查一次
0 * * * * /home/hpc/v100dd/v100dd12/code/3D-Mem/cursor/check_24h_timeout.sh

# 每天凌晨2点检查
0 2 * * * /home/hpc/v100dd/v100dd12/code/3D-Mem/cursor/check_24h_timeout.sh
```
