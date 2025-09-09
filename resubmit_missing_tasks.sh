#!/bin/bash

echo "=== Resubmitting missing tasks ==="

# Base directories
SCRIPT_BASE="/home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs"

# Resubmit missing tasks
echo "Resubmitting critique tasks..."
sbatch "$SCRIPT_BASE/critique/qwen-qfirst-top1-seed32.sh"
sbatch "$SCRIPT_BASE/critique/qwen-qfirst-top1-seed42.sh"
sbatch "$SCRIPT_BASE/critique/qwen-qfirst-top3-seed13.sh"
sbatch "$SCRIPT_BASE/critique/qwen-qfirst-top3-seed42.sh"
sbatch "$SCRIPT_BASE/critique/qwen-qfirst-top5-seed32.sh"
sbatch "$SCRIPT_BASE/critique/qwen-random-top3-seed19.sh"

echo "Resubmitting wo-critique tasks..."
sbatch "$SCRIPT_BASE/wo-critique/qwen-qfirst-top1-seed13.sh"
sbatch "$SCRIPT_BASE/wo-critique/qwen-qfirst-top1-seed32.sh"
sbatch "$SCRIPT_BASE/wo-critique/qwen-qfirst-top1-seed42.sh"
sbatch "$SCRIPT_BASE/wo-critique/qwen-qfirst-top3-seed13.sh"
sbatch "$SCRIPT_BASE/wo-critique/qwen-qfirst-top3-seed42.sh"
sbatch "$SCRIPT_BASE/wo-critique/qwen-random-top3-seed13.sh"

echo "=== All missing tasks submitted ==="
