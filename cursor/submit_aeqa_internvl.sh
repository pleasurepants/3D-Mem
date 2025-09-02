#!/bin/bash
# Submit InternVL runs for four seeds (each sbatch uses 2 GPUs; local vLLM per job)

set -euo pipefail

ROOT=/home/wiss/zhang/code/openeqa/3D-Mem
CURSOR_DIR=${ROOT}/cursor
SLURM_OUT=${ROOT}/slurm/internvl

mkdir -p ${SLURM_OUT}

for SEED in 19 32 568 82; do
  YAML=${CURSOR_DIR}/aeqa_internvl_seed${SEED}.yaml
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=i_2sl_${SEED}
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10-00:00:00
#SBATCH --exclude=worker-minor-1,worker-minor-3,worker-minor-4,worker-minor-5,worker-minor-6,worker-3,worker-4,worker-6,worker-7,worker-8,worker-9
#SBATCH --output=/home/wiss/zhang/code/openeqa/3D-Mem/slurm/internvl/2stage_listwise/2stage_listwise-${SEED}-%j.out
#SBATCH --partition all

echo "=== JOB START ==="; date; hostname; nvidia-smi; echo "SLURM_JOB_ID: \$SLURM_JOB_ID"
export LD_LIBRARY_PATH=/home/wiss/zhang/local_cuda118/cuda_cudart/targets/x86_64-linux/lib:$LD_LIBRARY_PATH

source /home/wiss/zhang/anaconda3/bin/activate vllm
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve OpenGVLab/InternVL3-8B \
  --served-model-name internvl \
  --port 8000 \
  --limit-mm-per-prompt '{"image": 20}' \
  --trust-remote-code &
VLLM_PID=\$!

echo "[INFO] Waiting for vLLM (internvl) server to be ready..."
for i in {1..300}; do
  if curl -s http://localhost:8000/v1/models > /dev/null; then echo "[INFO] internvl API is ready"; break; fi
  echo "  ... waiting (\$((i*2))s)"; sleep 2
  if [ \$i -eq 300 ]; then echo "[ERROR] Timeout: internvl server failed to start."; if [ -n "\$VLLM_PID" ] && kill -0 "\$VLLM_PID" 2>/dev/null; then kill "\$VLLM_PID"; fi; exit 1; fi
done

source /home/wiss/zhang/anaconda3/bin/activate 3dmem
cd ${ROOT}
source .env

CUDA_VISIBLE_DEVICES=1 python ${ROOT}/run_aeqa_evaluation_internvl.py -cf ${YAML} | cat

echo "[INFO] AEQA finished. Killing vLLM server (PID=\$VLLM_PID)..."
if [ -n "\$VLLM_PID" ] && kill -0 "\$VLLM_PID" 2>/dev/null; then kill "\$VLLM_PID"; else echo "[WARN] No running vLLM process to kill"; fi
echo "=== JOB END ==="
EOF
done
