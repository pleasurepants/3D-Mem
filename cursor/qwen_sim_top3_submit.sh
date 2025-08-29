#!/bin/bash
# Submit similarity top-3 runs for four seeds (with local vLLM server)

set -eo pipefail

ROOT=/home/hpc/v100dd/v100dd12/code/3D-Mem

for SEED in 13 568 82 19; do
  case $SEED in
    13)  SCRIPT=${ROOT}/script/alex/alex_qwen_seed13.sh ;;
    568) SCRIPT=${ROOT}/script/alex/alex_qwen_seed568.sh ;;
    82)  SCRIPT=${ROOT}/script/alex/alex_qwen_seed82.sh ;;
    19)  SCRIPT=${ROOT}/script/alex/alex_qwen_seed19.sh ;;
  esac

  echo "Submitting similarity top-3 for seed ${SEED} via ${SCRIPT}"
  sbatch "$SCRIPT"
done

echo "All similarity top-3 jobs submitted."


