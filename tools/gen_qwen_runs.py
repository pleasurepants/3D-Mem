import os
from pathlib import Path

BASE_YAML = """# General
seed: {seed}
exp_name: "seed{seed}"
output_parent_dir: "{out_dir}"
experience_filename: "{exp_file}"
scene_dataset_config_path: "/home/hpc/v100dd/v100dd12/code/3D-Mem/data/hm3d_annotated_basis.scene_dataset_config.json"
scene_data_path: "/anvme/workspace/v100dd12-3dmem/hm3d/data/3dmem"
questions_list_path: '/home/hpc/v100dd/v100dd12/code/3D-Mem/data/aeqa_questions-41.json'

concept_graph_config_path: "/home/hpc/v100dd/v100dd12/code/3D-Mem/cfg/concept_graph_default.yaml"

# major settings
choose_every_step: true
egocentric_views: true
prefiltering: true
top_k_categories: 10

# about detection model
yolo_model_name: /anvme/workspace/v100dd12-3dmem/model/yolov8x-world.pt
sam_model_name: /anvme/workspace/v100dd12-3dmem/model/sam_l.pt
class_set: scannet200

# about snapshots clustering
min_detection: 1

# camera, image
camera_height: 1.5
camera_tilt_deg: -30
img_width: 1280
img_height: 1280
hfov: 120

# whether to save visualization (which is slow)
save_visualization: true

# the image size for prompting gpt-4o
prompt_h: 360
prompt_w: 360

# navigation
num_step: 50
init_clearance: 0.3
extra_view_phase_1: 2
extra_view_angle_deg_phase_1: 60
extra_view_phase_2: 6
extra_view_angle_deg_phase_2: 40

# about tsdf, depth map, and frontier updates
explored_depth: 1.7
tsdf_grid_size: 0.1
margin_w_ratio: 0.25
margin_h_ratio: 0.6
planner:
  eps: 1
  max_dist_from_cur_phase_1: 1
  max_dist_from_cur_phase_2: 1
  final_observe_distance: 0.75
  surrounding_explored_radius: 0.7

  # about frontier selection
  frontier_edge_area_min: 4
  frontier_edge_area_max: 6
  frontier_area_min: 8
  frontier_area_max: 9
  min_frontier_area: 20
  min_frontier_area_layer0: 20
  min_frontier_area_layer1: 2
  max_frontier_angle_range_deg: 150
  region_equal_threshold: 0.95

# about scene graph construction
scene_graph:
  confidence: 0.003
  nms_threshold: 0.1
  iou_threshold: 0.5
  obj_include_dist: 3.5
  target_obj_iou_threshold: 0.6
"""

BASE_SH = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00 
#SBATCH --output=/home/hpc/v100dd/v100dd12/code/3D-Mem/slurm/experience/{variant}/{job}-%j.out
#SBATCH --partition a40

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

echo "[INFO] Starting vLLM (qwen) server on GPU 0..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate vllm
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
vllm serve /anvme/workspace/v100dd12-3dmem/model/Qwen2.5-VL-7B-Instruct \
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

echo "[INFO] Starting AEQA evaluation on GPU 1 (3dmem env)..."
source /home/hpc/v100dd/v100dd12/anaconda3/bin/activate 3dmem
source .env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

CUDA_VISIBLE_DEVICES=1 python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py \
    -cf {cfg_path} \
    --replay_mode {mode} \
    --replay_top {topk} \
    --retrieve_root /anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168

echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="
"""


def main():
    root = Path(__file__).resolve().parents[1]
    cfg_root = root / 'cfg' / 'alex_cfg' / 'qwen_runs'
    sh_root = root / 'script' / 'alex' / 'qwen_runs'

    variants = {
        'critique': 'experience_output.json',
        'wo-critique': 'experience_output_wo.json',
    }
    modes = ['sim', 'random']
    tops = [1, 3, 5]
    seeds = [13, 19, 32, 42]

    for variant, exp_file in variants.items():
        for mode in modes:
            for topk in tops:
                for seed in seeds:
                    # YAML
                    out_dir = f"/anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/experience/{variant}/{'random' if mode=='random' else 'sim'}-top{topk}"
                    y_dir = cfg_root / variant
                    y_dir.mkdir(parents=True, exist_ok=True)
                    y_name = f"qwen-{'random' if mode=='random' else 'sim'}-top{topk}-seed{seed}.yaml"
                    y_path = y_dir / y_name
                    y_content = BASE_YAML.format(seed=seed, out_dir=out_dir, exp_file=exp_file)
                    y_path.write_text(y_content)

                    # SH
                    s_dir = sh_root / variant
                    s_dir.mkdir(parents=True, exist_ok=True)
                    job = f"q-{ 'wo' if variant=='wo-critique' else 'crit' }-{'rand' if mode=='random' else 'sim'}-top{topk}-{seed}"
                    s_name = f"qwen-{'random' if mode=='random' else 'sim'}-top{topk}-seed{seed}.sh"
                    s_path = s_dir / s_name
                    cfg_path = str(y_path)
                    s_content = BASE_SH.format(job=job, variant=variant, cfg_path=cfg_path, mode=mode, topk=topk)
                    s_path.write_text(s_content)

    print('[gen] done')


if __name__ == '__main__':
    main()


