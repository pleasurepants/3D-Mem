#!/usr/bin/env bash
set -euo pipefail

ROOT_WS="/home/hpc/v100dd/v100dd12/code/3D-Mem"
CFG_BASE_TEMPLATE="${ROOT_WS}/cfg/alex_cfg/qwen_runs/hiera-cot/hiera-cot-seed32.yaml"

declare -A MODE_MAP=(
  [qfirst]="question-first"
  [sim]="sim"
  [random]="random"
)

SEEDS=(19 32 568)
MODES=(qfirst sim random)
TOPS=(1 3 5)

gen_yaml() {
  local root="$1" mode="$2" top="$3" seed="$4"
  local dst_dir="${ROOT_WS}/cfg/alex_cfg/qwen_runs/${root}/${mode}/top${top}"
  local dst_yaml="${dst_dir}/seed${seed}.yaml"
  local out_parent="/anvme/workspace/v100dd12-3dmem/openeqa/experience/${root}/${mode}/top${top}"
  mkdir -p "${dst_dir}"
  # 基于模板复制并修改三处字段：output_parent_dir、exp_name、seed
  sed \
    -e "s|^output_parent_dir:.*|output_parent_dir: \"${out_parent}\"|" \
    -e "s|^exp_name:.*|exp_name: \"seed${seed}\"|" \
    -e "s|^seed:.*|seed: 13|" \
    "${CFG_BASE_TEMPLATE}" > "${dst_yaml}"
}

gen_sh() {
  local root="$1" mode="$2" top="$3" seed="$4"
  local replay_mode="${MODE_MAP[${mode}]}"
  local cfg_path="${ROOT_WS}/cfg/alex_cfg/qwen_runs/${root}/${mode}/top${top}/seed${seed}.yaml"
  local dst_dir="${ROOT_WS}/script/alex/qwen_runs/${root}/${mode}/top${top}"
  local dst_sh="${dst_dir}/seed${seed}.sh"
  local log_dir="${ROOT_WS}/slurm/qwen/${root}/${mode}/top${top}"
  mkdir -p "${dst_dir}" "${log_dir}"
  cat > "${dst_sh}" << EOF
#!/bin/bash
#SBATCH --job-name=${root}-${mode}-top${top}-seed${seed}
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 
#SBATCH --output=${log_dir}/seed${seed}-%j.out
#SBATCH --partition a40

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

echo "=== JOB START ==="
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: \$SLURM_JOB_ID"

if [ -z "\$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[INFO] SLURM_JOB_GPUS not set, fallback to 0,1"
else
    export CUDA_VISIBLE_DEVICES=\$SLURM_JOB_GPUS
    echo "[INFO] CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES"
fi

echo "[INFO] Starting vLLM (qwen) server on GPU 0..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate vllm

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-VL-7B-Instruct \
    --served-model-name qwen \
    --port 8000 \
    --max-model-len 100000 \
    --limit-mm-per-prompt '{"image": 20}' &
VLLM_PID=\$!

echo "[INFO] Waiting for vLLM (qwen) server to be ready..."
for i in {1..300}; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "[INFO] ✅ qwen API is ready!"
        break
    fi
    echo "  ... waiting (\$((i*10))s)"
    sleep 10
    if [ \$i -eq 300 ]; then
        echo "[ERROR] ❌ Timeout: qwen server failed to start."
        if [ -n "\$VLLM_PID" ] && kill -0 "\$VLLM_PID" 2>/dev/null; then
            kill "\$VLLM_PID"
        fi
        exit 1
    fi
done

echo "[INFO] Starting AEQA evaluation on GPU 1 (3dmem env)..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source ${ROOT_WS}/.env
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 python ${ROOT_WS}/run_aeqa_evaluation_qwen.py \
    -cf ${cfg_path} \
    --replay_mode ${replay_mode} \
    --replay_top ${top} \
    --retrieve_root /anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168 \
    --chat_seed ${seed}

echo "[INFO] AEQA finished. Killing vLLM server (PID=\$VLLM_PID)..."
if [ -n "\$VLLM_PID" ] && kill -0 "\$VLLM_PID" 2>/dev/null; then
    kill "\$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="
EOF
  chmod +x "${dst_sh}"
}

main() {
  for root in cri-abs wo-cri-abs; do
    for mode in "${MODES[@]}"; do
      for top in "${TOPS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          gen_yaml "$root" "$mode" "$top" "$seed"
          gen_sh "$root" "$mode" "$top" "$seed"
        done
      done
    done
  done
}

main "$@"


