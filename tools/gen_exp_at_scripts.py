#!/usr/bin/env python3
import os
import re
from pathlib import Path

ROOT = Path('/home/hpc/v100dd/v100dd12/code/3D-Mem')
SCRIPT_ROOT = ROOT / 'script' / 'exp_at'
SLURM_ROOT = ROOT / 'slurm' / 'exp_at'
CFG_ROOT = ROOT / 'cfg' / 'exp_at'
BASE_CFG_PATH = ROOT / 'cfg' / 'alex_cfg' / 'eval_aeqa_debug.yaml'

# external resources (from qwen_debug.sh snippet)
RETRIEVE_ROOT = '/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set'
TRAJ_FILE = '/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/traj_abs_single.json'
EXP_TUPLE = '/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/exp_tuple_v0.json'

# output parent dir for these experiments
OUTPUT_PARENT_DIR = '/anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/exp_at'


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
    text = re.sub(r"^questions_list_path:\s*.*$", "questions_list_path: '/home/hpc/v100dd/v100dd12/code/3D-Mem/data/aeqa_questions-41.json'", text, flags=re.MULTILINE)
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
    cmd_head = 'CUDA_VISIBLE_DEVICES=1 python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py ' + bslash
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

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

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

export LD_LIBRARY_PATH=/home/hpc/v100dd/v100dd12/anaconda3/envs/iclblip/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

echo "[INFO] Starting vLLM (qwen) server on GPU 0..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate vllm

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vllm serve /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-VL-7B-Instruct     --served-model-name qwen     --port 8000     --max-model-len 100000     --limit-mm-per-prompt '{{"image": 20}}' &
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

echo "[INFO] Starting AEQA evaluation on GPU 1 (3dmem env)..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
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
"""


def main():
    base_cfg_text = load_base_cfg_text()

    stages = ['bvf', 'cvf']
    cats = [
        ('caption', 'cap'),
        ('caption-critique-abstraction', 'cca'),
        ('trajectory_experience', 'traj'),
    ]
    modes_map = {
        'caption': [('sim', 's'), ('random', 'r')],
        'caption-critique-abstraction': [('sim', 's'), ('random', 'r')],
        'trajectory_experience': [('traj_sim', 's'), ('traj_random', 'r')],
    }
    tops = [1, 3, 5]
    seeds = [13, 32, 568]

    for stage in stages:
        for cat_key, cat_short in cats:
            for mode_key, mode_short in modes_map[cat_key]:
                for top in tops:
                    top_dir = f'top-{top}'
                    # paths (top level only; no seed subfolder)
                    rel_dir = Path(stage) / cat_key / mode_key / top_dir
                    script_dir = SCRIPT_ROOT / rel_dir
                    slurm_dir = SLURM_ROOT / rel_dir
                    cfg_dir = CFG_ROOT / rel_dir
                    script_dir.mkdir(parents=True, exist_ok=True)
                    slurm_dir.mkdir(parents=True, exist_ok=True)
                    cfg_dir.mkdir(parents=True, exist_ok=True)

                    for seed in seeds:
                        base = f"{stage}-{cat_short}-{mode_short}-{top}-{seed}"
                        sh_path = script_dir / f"{base}.sh"
                        cfg_path = cfg_dir / f"{base}.yaml"

                        # write cfg (hierarchical exp_name with per-script seed suffix)
                        exp_name = f"{stage}/{cat_key}/{mode_key}/top-{top}/seed{seed}"
                        cfg_text = render_cfg(base_cfg_text, exp_name, seed)
                        with open(cfg_path, 'w', encoding='utf-8') as f:
                            f.write(cfg_text)

                        # write sh
                        sh_text = render_sh(stage, cat_key, cat_short, mode_key, mode_short, top, seed, cfg_path, slurm_dir)
                        with open(sh_path, 'w', encoding='utf-8') as f:
                            f.write(sh_text)
                        os.chmod(sh_path, 0o755)

    print(f"Generated scripts under: {SCRIPT_ROOT}")
    print(f"Generated cfgs under: {CFG_ROOT}")
    print(f"Generated slurm outputs folders under: {SLURM_ROOT}")


if __name__ == '__main__':
    main()


