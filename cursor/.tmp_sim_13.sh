#!/bin/bash
# Submit similarity top-5 runs for four seeds (with local vLLM server)

set -euo pipefail

ROOT=/home/hpc/v100dd/v100dd12/code/3D-Mem
CURSOR_DIR=${ROOT}/cursor

for SEED in 13 568 82 19; do
  YAML=${CURSOR_DIR}/qwen_${SEED}_sim_top5.yaml
  sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=q-ee-sim-top5-13
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/ee/qwen-sim-top5-13-%j.out
#SBATCH --partition a40

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
echo "=== JOB START ==="; date; hostname; nvidia-smi; echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unknown}"

# Start vLLM server (GPU 0)
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate vllm
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen \
  --port 8000 \
  --max-model-len 100000 \
  --limit-mm-per-prompt '{"image": 20}' &
VLLM_PID=
VLLM_PID=$!

echo "[INFO] Waiting for vLLM (qwen) server to be ready..."
for i in {1..300}; do
  if curl -s http://localhost:8000/v1/models > /dev/null; then echo "[INFO] qwen API is ready"; break; fi
  echo "  ... waiting ($((i*10))s)"; sleep 10
  if [ $i -eq 300 ]; then echo "[ERROR] Timeout: qwen server failed to start."; if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID"; fi; exit 1; fi
done

# Run evaluation (GPU 1)
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source /home/hpc/v100dd/v100dd12/code/3D-Mem/.env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py \
  -cf /home/hpc/v100dd/v100dd12/code/3D-Mem/cursor/qwen_13_sim_top5.yaml \
  --replay_mode sim \
  --replay_top 5 | cat

echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID"; else echo "[WARN] No running vLLM process to kill"; fi
echo "=== JOB END ==="
EOF
  # Replace placeholders and submit
  sed -e "s/13/${SEED}/g" -e "s#qwen_13_sim_top5.yaml#qwen_${SEED}_sim_top5.yaml#g" ${CURSOR_DIR}/qwen_sim_top5_submit.sh > ${CURSOR_DIR}/.tmp_sim_${SEED}.sh
  sbatch ${CURSOR_DIR}/.tmp_sim_${SEED}.sh | cat
  rm -f ${CURSOR_DIR}/.tmp_sim_${SEED}.sh
done


