#!/bin/bash
#SBATCH --job-name=k6-32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/qwen/hierarchy/kmeans_6/k6-32-%j.out
#SBATCH --partition a100

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "=== JOB START ==="; date; hostname; nvidia-smi

source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate vllm
CUDA_VISIBLE_DEVICES=0 vllm serve /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 100000 \
  --limit-mm-per-prompt '{"image": 20}' \
  --trust-remote-code &

for i in {1..120}; do
  if curl -s http://localhost:8000/v1/models > /dev/null; then echo "[INFO] qwen ready"; break; fi
  sleep 5
  if [ $i -eq 120 ]; then echo "[ERROR] qwen timeout"; exit 1; fi
done

source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source /home/hpc/v100dd/v100dd12/code/3D-Mem/.env

CUDA_VISIBLE_DEVICES=1 python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py \
  -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/script/layer2_mode/kmeans_6/qwen_32.yaml \
  --hierarchy_mode True \
  --layer2_num 2 \
  --chat_seed 32

echo "=== JOB END ==="
