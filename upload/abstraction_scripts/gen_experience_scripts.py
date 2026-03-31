import os
import re

ROOT = "./"
SCR = os.path.join(ROOT, "script/experience")
CFG = os.path.join(ROOT, "cfg/experience")
SLURM = os.path.join(ROOT, "slurm/experience")
TEMPLATE_CFG = os.path.join(ROOT, "cfg/experience/baseline/b-13.yaml")

ABS_QUESTION_JSON = "./data/aeqa_questions-41.json"
RETRIEVE_ROOT = "/path/to/workspace/openeqa/ee_qwen/qwen-exp-168"
EXP_TUPLE = f"{RETRIEVE_ROOT}/exp_tuple_v0.json"

TYPE_SPECS = [
    ("caption", {"caption": "true", "critique": "false", "abstraction": "false"}, "c"),
    (
        "caption-critique",
        {"caption": "true", "critique": "true", "abstraction": "false"},
        "cc",
    ),
    (
        "caption-abstraction",
        {"caption": "true", "critique": "false", "abstraction": "true"},
        "ca",
    ),
    (
        "caption-critique-abstraction",
        {"caption": "true", "critique": "true", "abstraction": "true"},
        "cca",
    ),
]

MODES = [("sim", "s"), ("random", "r")]
TOPS = [1, 3, 5]
SEEDS = [32, 568, 69]


def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def read_template() -> str:
    with open(TEMPLATE_CFG, "r", encoding="utf-8") as f:
        return f.read()


def make_cfg_text(template: str, exp_name: str) -> str:
    text = re.sub(r'^exp_name: ".*"', f'exp_name: "{exp_name}"', template, flags=re.M)
    text = re.sub(
        r"questions_list_path: '.*/aeqa_questions-41.json'",
        f"questions_list_path: '{ABS_QUESTION_JSON}'",
        text,
        flags=re.M,
    )
    return text


def write_cfg(type_name: str, mode: str, top: int, job: str, template: str) -> str:
    cfg_dir = os.path.join(CFG, type_name, mode, f"top-{top}")
    ensure_dirs(cfg_dir)
    cfg_path = os.path.join(cfg_dir, f"{job}.yaml")
    cfg_text = make_cfg_text(template, f"{type_name}/{mode}/top-{top}/{job}")
    with open(cfg_path, "w", encoding="utf-8") as f:
        f.write(cfg_text)
    return cfg_path


def write_sh(
    type_name: str,
    type_short: str,
    mode: str,
    mode_short: str,
    top: int,
    seed: int,
    flags: dict,
    cfg_path: str,
):
    sh_dir = os.path.join(SCR, type_name, mode, f"top-{top}")
    slurm_dir = os.path.join(SLURM, type_name, mode, f"top-{top}")
    ensure_dirs(sh_dir, slurm_dir)
    job = f"{type_short}-{mode_short}-{top}-{seed}"
    sh_path = os.path.join(sh_dir, f"{job}.sh")
    out_path = os.path.join(slurm_dir, f"{job}-%j.out")
    # choose GPU/partition
    if type_name in {"caption-critique", "caption-abstraction"}:
        gres_gpu = "a100:2"
        partition = "a100"
    else:
        gres_gpu = "a40:2"
        partition = "a40"

    sh = f"""#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --nodes=1
#SBATCH --gres=gpu:{gres_gpu}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 
#SBATCH --output={out_path}
#SBATCH --partition {partition}

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

CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve /path/to/workspace/model/Qwen2.5-VL-7B-Instruct \
    --served-model-name qwen \
    --port 8000 \
    --max-model-len 100000 \
    --limit-mm-per-prompt '{{"image": 20}}' &
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

CUDA_VISIBLE_DEVICES=1 python {ROOT}/run_aeqa_evaluation_qwen.py \
    -cf {cfg_path} \
    --replay_mode {mode} \
    --replay_top {top} \
    --retrieve_root {RETRIEVE_ROOT} \
    --exp_tuple {EXP_TUPLE} \
    --chat_seed {seed} \
    --caption {flags['caption']} \
    --critique {flags['critique']} \
    --abstraction {flags['abstraction']}

echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="