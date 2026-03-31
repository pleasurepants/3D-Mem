#!/usr/bin/env python3
import os
import re
from pathlib import Path

ROOT = Path('./')
SCRIPT_ROOT = ROOT / 'script' / 'exp_at'
SLURM_ROOT = ROOT / 'slurm' / 'exp_at'
CFG_ROOT = ROOT / 'cfg' / 'exp_at'
BASE_CFG_PATH = ROOT / 'cfg' / 'alex_cfg' / 'eval_aeqa_debug.yaml'

# external resources (from qwen_debug.sh snippet)
RETRIEVE_ROOT = '/path/to/workspace/openeqa/pipeline_2/experience_set'
TRAJ_FILE = '/path/to/workspace/openeqa/pipeline_2/experience_set/traj_abs_single.json'
EXP_TUPLE = '/path/to/workspace/openeqa/pipeline_2/experience_set/exp_tuple_v0.json'

# output parent dir for these experiments
OUTPUT_PARENT_DIR = '/path/to/workspace/openeqa/pipeline_2/exp_at'


def load_base_cfg_text() -> str:
    with open(BASE_CFG_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def render_cfg(base_text: str, exp_name: str, seed_value: int) -> str:
    text = base_text
    # replace exp_name
    text = re.sub(r'^exp_name:\s*".*?"\s*$', f'exp_name: "{exp_name}"', text, flags=re.MULTILINE)
    # replace output_parent_dir
    text = re.sub(r'^output_parent_dir:\s*".*?"\s*$', f'output_parent_dir: "{OUTPUT_PARENT_DIR}"', text, flags=re.MULTILINE)
    # seed is fixed to 13 regardless of chat_seed used in scripts
    text = re.sub(r'^seed:\s*\d+\s*$', 'seed: 13', text, flags=re.MULTILINE)
    # override questions_list_path
    text = re.sub(r"^questions_list_path:\s*.*$", "questions_list_path: './data/aeqa_questions-41.json'", text, flags=re.MULTILINE)
    return text


def render_sh(stage: str, cat_key: str, cat_short: str, mode_key: str, mode_short: str, top: int, seed: int, cfg_path: Path, slurm_out_dir: Path) -> str:
    base = f"{stage}-{cat_short}-{mode_short}-{top}-{seed}"
    job_name = base
    slurm_out = slurm_out_dir / f"{base}-%j.out"
    # booleans per category
    if cat_key == 'caption':
        caption, critique, abstraction = True, False, False
    elif cat_key == 'caption-critique-abstraction':
        caption, critique, abstraction = True, True, True
    else:  # trajectory_experience
        caption, critique, abstraction = True, True, True

    caption_s = 'true' if caption else 'false'
    critique_s = 'true' if critique else 'false'
    abstraction_s = 'true' if abstraction else 'false'

    # Build replay_mode
    replay_mode = mode_key

    # Build multi-line python command (one arg per line, with continuation backslashes)
    bslash = chr(92)
    cmd_head = 'CUDA_VISIBLE_DEVICES=1 python ./run_aeqa_evaluation_qwen.py ' + bslash
    args_core = [
        f'  -cf {cfg_path}',
        f'  --replay_mode {replay_mode}',
        f'  --replay_top {top}',
        f'  --use_episodic_context 1',
        f'  --retrieve_root {RETRIEVE_ROOT}',
    ]
    if mode_key.startswith('traj_'):
        args_core.append(f'  --traj_file {TRAJ_FILE}')
    args_core.extend([
        f'  --exp_tuple {EXP_TUPLE}',
        f'  --exp_at {stage}',
        f'  --chat_seed {seed}',
        f'  --caption {caption_s}',
        f'  --critique {critique_s}',
        f'  --abstraction {abstraction_s}',  # last line: no trailing backslash
    ])
    # add trailing backslash to all but last
    if args_core:
        args_lines = [*(line + ' ' + bslash for line in args_core[:-1]), args_core[-1]]
    else:
        args_lines = []
    py_cmd_str = '\n'.join([cmd_head] + args_lines)

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output={slurm_out}
#SBATCH --partition a100

export LD_LIBRARY_PATH=/path/to/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY
echo "=== JOB START ==="
date
hostname
nvidia-smi
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

if [ -z "$SLURM_JOB_GPUS" ]; then
    export CUDA_VISIBLE_DEVICES=0,1
    echo "[INFO] SLURM_JOB_GPUS not set, fallback to 0,1"
else
    export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS
    echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

export LD_LIBRARY_PATH=/path/to/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

echo "[INFO] Starting vLLM (qwen) server on GPU 0..."
source /path/to/anaconda3/bin/activate vllm

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve /path/to/workspace/model/Qwen2.5-VL-7B-Instruct     --served-model-name qwen     --port 8000     --max-model-len 100000     --limit-mm-per-prompt '{{"image": 20}}' &
VLLM_PID=$!

echo "[INFO] Waiting for vLLM (qwen) server to be ready..."
for i in {{1..300}}; do
    if curl -s http://localhost:8000/v1/models > /dev/null; then
        echo "[INFO] ✅ qwen API is ready!"
        break
    fi
    echo "  ... waiting ($((i*10))s)"
    sleep 10
    if [ $i -eq 300 ]; then
        echo "[ERROR] ❌ Timeout: qwen server failed to start."
        if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
            kill "$VLLM_PID"
        fi
        exit 1
    fi
done

echo "[INFO] Starting AEQA evaluation on GPU 1 (reexplore env)..."
source /path/to/anaconda3/bin/activate reexplore
source .env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

{py_cmd_str}

echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="