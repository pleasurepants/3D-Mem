#!/bin/bash

echo "=== Quick resubmit of remaining 12 failed tasks ==="

# Submit the remaining tasks
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/critique/qwen-qfirst-top1-seed32.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/critique/qwen-qfirst-top1-seed42.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/critique/qwen-qfirst-top3-seed13.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/critique/qwen-qfirst-top3-seed42.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/critique/qwen-qfirst-top5-seed32.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/critique/qwen-random-top3-seed19.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/wo-critique/qwen-qfirst-top1-seed13.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/wo-critique/qwen-qfirst-top1-seed32.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/wo-critique/qwen-qfirst-top1-seed42.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/wo-critique/qwen-qfirst-top3-seed13.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/wo-critique/qwen-qfirst-top3-seed42.sh
sbatch /home/hpc/v100dd/v100dd12/code/3D-Mem/script/alex/qwen_runs/wo-critique/qwen-random-top3-seed13.sh

echo "=== All 12 tasks submitted ==="
