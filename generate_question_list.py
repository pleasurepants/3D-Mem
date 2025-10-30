from omegaconf import OmegaConf
import os
import json
from tqdm import tqdm

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

import sys
sys.path.append("/nfs/data8/jingpei/eqa/3D-Mem")

from src.goatbench_utils import prepare_goatbench_navigation_goals
from src.utils import get_pts_angle_goatbench
from src.tsdf_planner_hdbscan import TSDFPlanner
from src.scene_goatbench import Scene
from src.logger_goatbench import Logger
import open_clip
from ultralytics import YOLOWorld, SAM
import logging
from src.geom import get_scene_bnds

# output_dir = "/nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/"
# test_data_dir = "/nfs/data8/jingpei/eqa/3D-Mem/data/goat_bench/val_unseen/content"
# scene_data_path = "/nfs/data8/jingpei/eqa/open-eqa/data/scene_datasets/hm3d/"
cfg_file = "/nfs/data8/jingpei/eqa/3D-Mem/cfg/dbs_cfg/goatbench_qwen_rollout.yaml"
start_ratio = 0.0
end_ratio = 1.0
split = 1

CLIP_PATH = '/nfs/data8/jingpei/eqa/models/CLIP-ViT-H-14-laion2B-s32B-b79K'

cfg = OmegaConf.load(cfg_file)
OmegaConf.resolve(cfg)
cfg.output_parent_dir = cfg.output_parent_dir + "_log"

cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
if not os.path.exists(cfg.output_dir):
    os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
        
cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
OmegaConf.resolve(cfg_cg)

question_list = []

scene_data_list = os.listdir(cfg.test_data_dir)
# use training set to build bank
if 'train' in cfg.test_data_dir:
    all_scene_ids = os.listdir(cfg.scene_data_path + "/train")
else:
    all_scene_ids = os.listdir(cfg.scene_data_path + "/val")

detection_model = YOLOWorld(cfg.yolo_model_name)
logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    # "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    "ViT-H-14", pretrained=CLIP_PATH + "/open_clip_pytorch_model.bin"
)
# clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
logging.info(f"Load CLIP model successful!")

logger = Logger(
    cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
)

for scene_data_file in tqdm(scene_data_list):
    scene_name = scene_data_file.split(".")[0]
    scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
    scene_data = json.load(
        open(os.path.join(cfg.test_data_dir, scene_data_file), "r")
    )

    # selecat the episodes according to the split
    ## why only one episode -> setting in 3d-mem
    scene_data["episodes"] = scene_data["episodes"][split - 1 : split]
    total_episodes = len(scene_data["episodes"])
    all_navigation_goals = scene_data[
        "goals"
    ]  # obj_id to obj_data, apply for all episodes in this scene

    for episode_idx, episode in enumerate(scene_data["episodes"]):
        episode_id = episode["episode_id"]
        all_subtask_goal_types, all_subtask_goals = (
            prepare_goatbench_navigation_goals(
                scene_name=scene_name,
                episode=episode,
                all_navigation_goals=all_navigation_goals,
            )
        )
        # all_subtask_goals[0/1/2][0].keys(): ['object_category', 'object_id', 'position', 'view_points', 'children_object_categories', 'lang_desc', 'image_goals']
        # first index for subtask, second index for goals (len=1 for type description and image, len as in goals for type object)
        
        pts, angle = get_pts_angle_goatbench(
            episode["start_position"], episode["start_rotation"]
        )
        
        scene = Scene(
            scene_id=scene_id,
            cfg=cfg,
            graph_cfg=cfg_cg,
            detection_model=detection_model,
            sam_predictor=sam_predictor,
            clip_model=clip_model,
            clip_preprocess=clip_preprocess,
            clip_tokenizer=clip_tokenizer,
        )
        floor_height = pts[1]
        tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height=floor_height,
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=cfg.init_clearance * 2,
        )
        
        for subtask_idx, (goal_type, subtask_goal) in enumerate(
            zip(all_subtask_goal_types, all_subtask_goals)
        ):
            subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
            
            subtask_metadata = logger.init_subtask(
                subtask_id=subtask_id,
                goal_type=goal_type,
                subtask_goal=subtask_goal,
                pts=pts,
                scene=scene,
                tsdf_planner=tsdf_planner,
            )

            logging.info(f"Subtask {subtask_id} initialization successful!")
            question_list.append(subtask_metadata)
            
with open(os.path.join(cfg.output_dir, "question_list.json"), "w") as f:
    json.dump(question_list, f, indent=4)