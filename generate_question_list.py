import os, json, re
from typing import Optional, Dict, List
import uuid

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"

SENTENCE_TRANSFORMERS_PATH = '/nfs/data8/jingpei/eqa/models/clip-ViT-B-32'
CLIP_PATH = '/nfs/data8/jingpei/eqa/models/CLIP-ViT-H-14-laion2B-s32B-b79K'

import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import math
import time
import json
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
# from src.tsdf_planner import TSDFPlanner, Frontier, SnapShot
from src.tsdf_planner_hdbscan import TSDFPlanner, Frontier, SnapShot
from src.scene_goatbench import Scene
from src.utils import resize_image, calc_agent_subtask_distance, get_pts_angle_goatbench
from src.goatbench_utils import prepare_goatbench_navigation_goals
# from src.query_vlm_goatbench import query_vlm_for_response
from src.query_vlm_goatbench_qwen import query_vlm_for_response
from src.logger_goatbench import Logger

def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)

def main(cfg, start_ratio=0.0, end_ratio=1.0, split=1):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    scene_data_list = os.listdir(cfg.test_data_dir)
    num_scene = len(scene_data_list)
    # random.shuffle(scene_data_list)   # consistent for debugging

    # split the test data by scene
    scene_data_list = scene_data_list[
        int(start_ratio * num_scene) : int(end_ratio * num_scene)
    ]
    num_episode = 0
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), "r") as f:
            num_episode += len(json.load(f)["episodes"])
    logging.info(
        f"Total number of episodes: {num_episode}; Selected episodes: {len(scene_data_list)}"
    )
    logging.info(f"Total number of scenes: {len(scene_data_list)}")

    # all_scene_ids = os.listdir(cfg.scene_data_path + "/train") + os.listdir(
    #     cfg.scene_data_path + "/val"
    # )

    if 'train' in cfg.test_data_dir:
        all_scene_ids = os.listdir(cfg.scene_data_path + "/train")
    else:
        all_scene_ids = os.listdir(cfg.scene_data_path + "/val")



    # load detection and segmentation models
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

    # Initialize the logger
    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
    )


    question_list = []
    for scene_data_file in scene_data_list:
        # load goatbench data
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
            # logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}, episode {episode_idx + 1}/{total_episodes}")
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

            # check whether this episode has been processed
            finished_subtask_ids = list(logger.success_by_snapshot.keys())
            finished_episode_subtask = [
                subtask_id
                for subtask_id in finished_subtask_ids
                if subtask_id.startswith(f"{scene_id}_{episode_id}_")
            ]
            if len(finished_episode_subtask) >= len(all_subtask_goals):
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            pts, angle = get_pts_angle_goatbench(
                episode["start_position"], episode["start_rotation"]
            )

            # load scene
            try:
                del scene
            except:
                pass
            scene = Scene(
                scene_id,
                cfg,
                cfg_cg,
                detection_model,
                sam_predictor,
                clip_model,
                clip_preprocess,
                clip_tokenizer,
            )

            # initialize the TSDF
            floor_height = pts[1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height=floor_height,
                floor_height_offset=0,
                pts_init=pts,
                init_clearance=cfg.init_clearance * 2,
                save_visualization=cfg.save_visualization,
            )

            episode_dir, eps_frontier_dir, eps_snapshot_dir = logger.init_episode(
                episode_id=f"{scene_id}_ep_{episode_id}"
            )   # diff w.r.t. aeqa: init_pts_voxel init in init_subtask(), no eps_chosen_snapshot_dir

            logging.info(f"\n\nScene {scene_id} initialization successful!")
            
            ## lifelong-memory
            lifelong_json_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, "lifelong_storage.json")

            # run questions in the scene
            global_step = -1    # in the whole episode
            for subtask_idx, (goal_type, subtask_goal) in enumerate(
                zip(all_subtask_goal_types, all_subtask_goals)
            ):
                subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"
                logging.info(
                    f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}"
                )

                subtask_metadata = logger.init_subtask(
                    subtask_id=subtask_id,
                    goal_type=goal_type,
                    subtask_goal=subtask_goal,
                    pts=pts,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                )

                question_list.append(subtask_metadata)
    
    json.dump(
        question_list,
        open(cfg.output_dir + "/question_list.json", "w"),
        indent=4,
        default=_json_default,
    )


if __name__ == "__main__":
    # Get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    parser.add_argument("--split", help="which episode", default=1, type=int)
    parser.add_argument("--replay_mode", help="replay selection mode: sim or random", default="sim", type=str)
    parser.add_argument("--replay_top", help="top-k for replay candidates", default=1, type=int)
    parser.add_argument("--retrieve_root", help="external retrieve root; expects replay_step_info.json & experience_output.json inside", default="", type=str)
    parser.add_argument("--use_episodic_context", help="whether to enable episodic context (0/1)", default=1, type=int)
    parser.add_argument("--chat_seed", help="random seed for vLLM generation (decoupled from cfg.seed)", default=None, type=int)
    # not used here, but `vllm serve ... --seed 0`
    parser.add_argument("--exp_tuple", help="path to exp_tuple json for EXPERIENCE REPLAY (no default)", default="", type=str)
    # toggles for injecting experience/critique/abstraction from JSON (default off)
    _bool = lambda x: str(x).lower() in ("1", "true", "t", "yes", "y")
    parser.add_argument("--caption", "--experience", dest="caption", help="inject base caption tuple lines", default=False, type=_bool)
    parser.add_argument("--critique", help="inject critique reflection lines", default=False, type=_bool)
    parser.add_argument("--abstraction", help="inject abstraction guideline lines", default=False, type=_bool)
    parser.add_argument("--traj_file", help="trajectory json for traj_* modes (qid -> {question, abstraction, thinking_process})", default="", type=str)
    # replay injection stage control
    # preferred flag: --exp_at; aliases: --replay_at / --inject_stage for backward compatibility
    parser.add_argument("--exp_at", "--replay_at", "--inject_stage", dest="exp_at", help="limit replay injection stage: '' (default, both), 'bvf' (layer0 only), 'cvf' (layer1 only)", default="", type=str)
    # ppl_rank mode parameter
    parser.add_argument("--ppl_rank", help="perplexity rank category for filtering: 'low', 'medium', or 'high' (case-insensitive)", default="", type=str)
    parser.add_argument("--ppl_rank_file", help="path to ppl_rank json file", default="/home/hpc/v100dd/v100dd12/code/3D-Mem/perplexity/traj_abs_format_ppl_rank.json", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    cfg.output_parent_dir = cfg.output_parent_dir + "_log"

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir),
        f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}_{args.split}.log",
    )

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

    # Set up the logging format
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)
        
    # ppl_rank logging (moved after logging setup)
    if args.ppl_rank:
        logging.info(f"[PPL_RANK] Mode enabled: category={cfg.ppl_rank}, file={cfg.ppl_rank_file}")

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio, split=args.split)


"""
output_dir: output_parent_dir/exp_name/
episode_dir: output_dir/episode_id/, episode_id = {scene_id}_ep_{episode_id}
eps_frontier_dir: episode_dir/frontier/
eps_snapshot_dir: episode_dir/snapshot/
subtask_object_observe_dir: output_dir/subtask_id/object_observations/
"""