#!/bin/bash

# Script to resubmit the remaining 12 failed tasks that are not in the queue
echo "=== Resubmitting remaining 12 failed tasks ==="

SCRIPT_BASE="/home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs"

# List of remaining tasks to resubmit
tasks=(
    "critique/qwen-qfirst-top1-seed32.sh"
    "critique/qwen-qfirst-top1-seed42.sh"
    "critique/qwen-qfirst-top3-seed13.sh"
    "critique/qwen-qfirst-top3-seed42.sh"
    "critique/qwen-qfirst-top5-seed32.sh"
    "critique/qwen-random-top3-seed19.sh"
    "wo-critique/qwen-qfirst-top1-seed13.sh"
    "wo-critique/qwen-qfirst-top1-seed32.sh"
    "wo-critique/qwen-qfirst-top1-seed42.sh"
    "wo-critique/qwen-qfirst-top3-seed13.sh"
    "wo-critique/qwen-qfirst-top3-seed42.sh"
    "wo-critique/qwen-random-top3-seed13.sh"
)

submitted_count=0
echo "Found ${#tasks[@]} tasks to resubmit"

for task_script in "${tasks[@]}"; do
    script_path="$SCRIPT_BASE/$task_script"

    if [[ -f "$script_path" ]]; then
        echo "Submitting: $task_script"
        if sbatch "$script_path"; then
            ((submitted_count++))
            echo "  ✅ Successfully submitted"
        else
            echo "  ❌ Failed to submit"
        fi
    else
        echo "  ⚠️  Script not found: $script_path"
    fi

    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo "=== Resubmission complete ==="
echo "Successfully submitted $submitted_count out of ${#tasks[@]} tasks"

# Show current queue status
echo ""
echo "=== Current queue status ==="
squeue -u v100dd12 | head -10
