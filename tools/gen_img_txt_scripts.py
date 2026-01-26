#!/usr/bin/env python3
"""
生成 img 和 txt 模式的脚本和配置文件
"""
import os

BASE_DIR = "/home/hpc/v100dd/v100dd12/code/3D-Mem"
SLURM_OUT_DIR = f"{BASE_DIR}/slurm/rebuttal"
SCRIPT_DIR = f"{BASE_DIR}/script/rebuttal"
CFG_DIR = f"{BASE_DIR}/cfg/rebuttal"
OUTPUT_PARENT_DIR = "/anvme/workspace/v100dd12-3dmem_rebuttal"

MODES = ["img", "txt"]
TOP_KS = [1, 3, 5]
SEEDS = [13, 32, 568]


def generate_yaml(mode, top_k, seed):
    exp_name = f"{mode}/top-{top_k}/s{seed}"
    content = f'''# General
seed: {seed}
exp_name: "{exp_name}"
output_parent_dir: "{OUTPUT_PARENT_DIR}"
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
'''
    return content


def generate_shell(mode, top_k, seed):
    job_name = f"{mode}-t{top_k}-s{seed}"
    slurm_out = f"{SLURM_OUT_DIR}/{mode}/{job_name}-%j.out"
    cfg_path = f"{CFG_DIR}/{mode}/top-{top_k}/s{seed}.yaml"
    
    content = f'''#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --gres=gpu:a40:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00 
#SBATCH --output={slurm_out}
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

CUDA_VISIBLE_DEVICES=1 python /home/hpc/v100dd/v100dd12/code/3D-Mem/run_aeqa_evaluation_qwen.py -cf {cfg_path} --replay_mode {mode} --replay_top {top_k} --retrieve_root /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set --chat_seed {seed} --traj_file /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/captions_reflection_abstraction_v1.json --caption true --critique true --abstraction true --exp_tuple /anvme/workspace/v100dd12-3dmem/openeqa/pipeline_2/training_set/exp_tuple_v0.json

echo "[INFO] AEQA finished. Killing vLLM server (PID=$VLLM_PID)..."
if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID"
else
    echo "[WARN] No running vLLM process to kill"
fi

echo "=== JOB END ==="
'''
    return content


def main():
    for mode in MODES:
        for top_k in TOP_KS:
            for seed in SEEDS:
                # Generate YAML
                yaml_dir = f"{CFG_DIR}/{mode}/top-{top_k}"
                yaml_path = f"{yaml_dir}/s{seed}.yaml"
                os.makedirs(yaml_dir, exist_ok=True)
                with open(yaml_path, 'w') as f:
                    f.write(generate_yaml(mode, top_k, seed))
                print(f"Created: {yaml_path}")
                
                # Generate Shell script
                script_dir = f"{SCRIPT_DIR}/{mode}/top-{top_k}"
                script_path = f"{script_dir}/{mode}-t{top_k}-s{seed}.sh"
                os.makedirs(script_dir, exist_ok=True)
                with open(script_path, 'w') as f:
                    f.write(generate_shell(mode, top_k, seed))
                os.chmod(script_path, 0o755)
                print(f"Created: {script_path}")


if __name__ == "__main__":
    main()
