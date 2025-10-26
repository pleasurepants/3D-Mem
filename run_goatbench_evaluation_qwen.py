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


def question_room(folder_path, questions_list_path):
    pass


def check_lifelong_memory(lifelong_json_path, lifelong_memory, cfg, question):
    pass


def tuple_step_save(
    tuple_save_path: str,
    question_id: str,   # f"{scene_id}_{episode_id}_{subtask_idx}"
    question: str,
    cnt_step: str,  # task-{}_step-{}, consistent to enable retrieval
    cfg,
    lifelong_json_path: str,
    subtask_metadata: dict,
    # question_data: Optional[dict] = None,   # may carry {"episode_history": "..."}
    # episode_history_id: Optional[str] = None,  # explicit episode id, if known
    final_reward: Optional[str] = None  # 'pass' or 'fail'
):
    # ---------- load or init ----------
    if os.path.exists(tuple_save_path):
        with open(tuple_save_path, 'r', encoding='utf-8') as f:
            saved_result = json.load(f)
    else:
        saved_result = {}

    def is_new_schema(data: dict) -> bool:
        # Heuristic: new schema has top-level episodes (strings) each mapping to dicts of question_ids
        # Old schema had top-level question_ids mapping to {"question":..., "steps":...}
        if not data:
            return True
        # If any top-level value is a dict whose keys look like "step_*" or has "question" -> likely old
        for k, v in data.items():
            if isinstance(v, dict) and ("steps" in v or "question" in v):
                return False
        return True

    def migrate_old_schema_if_needed(data: dict) -> dict:
        if not data:
            return data
        if is_new_schema(data):
            return data
        # Move all top-level question_ids under a new generated episode
        new_episode = f"episode-{uuid.uuid4().hex[:8]}"
        migrated = {new_episode: {}}
        for qid, qcontent in data.items():
            migrated[new_episode][qid] = qcontent
            # drop any leftover "episode_history" key if present
            if isinstance(qcontent, dict) and "episode_history" in qcontent:
                qcontent.pop("episode_history", None)
        return migrated

    saved_result = migrate_old_schema_if_needed(saved_result)

    # ---------- resolve episode_id for this question ----------
    def find_existing_episode_for_question(data: dict, qid: str) -> Optional[str]:
        for ep_id, ep_bucket in data.items():
            if isinstance(ep_bucket, dict) and qid in ep_bucket:
                return ep_id
        return None

    existing_ep = find_existing_episode_for_question(saved_result, question_id)
    
    """
    {
        "{episode_id}": {
            "{question_id}": {
                "question": "{question}",
                "steps": {
                    "{step_key}": {
                        "frontier": {
                            "{layer0_map}"
                        },
                        "chosen_frontier": {
                            "layer0": "{chosen_l0_path}",
                            "layer1": "{chosen_l1_path}"
                        }
                        "memory_snapshots": {
                            "{img_name}": "{obj_list}"
                        }
                    }
                },
                "final_reward": "{final_reward}"
            },
        }
    }
    """

    # priority: existing > explicit param > question_data > auto-generate
    # ep_id = (
    #     existing_ep
    #     or episode_history_id
    #     or (question_data.get("episode_history") if question_data else None)
    #     or f"episode-{uuid.uuid4().hex[:8]}"
    # )
    # ep_id = question_id.split("_")[0]
    ep_id, ep_sub_idx, subtask_idx = question_id.split("_") # ep_id = scene_id

    if ep_id not in saved_result:
        saved_result[ep_id] = {}

    # init question bucket under this episode
    if question_id not in saved_result[ep_id]:
        saved_result[ep_id][question_id] = {"question": question, "steps": {}}
    else:
        # keep existing question text if already stored; otherwise set it
        saved_result[ep_id][question_id].setdefault("question", question)
        saved_result[ep_id][question_id].setdefault("steps", {})
        # saved_result[ep_id][question_id].setdefault("subtask_metadata", subtask_metadata)

    # step_key = f"step_{cnt_step}"
    step_key = f"step_{cnt_step.split('-')[-1]}"    # question_id include subtask_idx
    saved_result[ep_id][question_id]["steps"][step_key] = {}

    # --- dirs ---
    # q_root = os.path.join(cfg.output_parent_dir, cfg.exp_name, question_id)
    q_root = os.path.join(cfg.output_parent_dir, cfg.exp_name, f"{ep_id}_ep_{ep_sub_idx}")
    frontier_dir = os.path.join(q_root, 'frontier')
    chosen_dir = os.path.join(q_root, 'chosen_frontier')

    def rel_frontier_path(fname: str) -> str:
        return f"frontier/{fname}"

    # -------- frontier (two layers) -> flattened mapping --------
    layer0_prefix = f"{cnt_step}-layer0-"
    layer1_prefix = f"{cnt_step}-layer1-"

    layer0_files: List[str] = []
    layer1_files: List[str] = []

    if os.path.exists(frontier_dir):
        for fn in os.listdir(frontier_dir):
            if fn.endswith(".png"):
                if fn.startswith(layer0_prefix):
                    layer0_files.append(fn)
                elif fn.startswith(layer1_prefix):
                    layer1_files.append(fn)

    layer0_files.sort()
    layer1_files.sort()

    layer0_map: Dict[str, List[str]] = {}
    for l0 in layer0_files:
        m = re.match(rf"^{cnt_step}-layer0-(\d+)\.png$", l0)
        if not m:
            continue
        x = m.group(1)
        children = [
            rel_frontier_path(fn)
            for fn in layer1_files
            if fn.startswith(f"{cnt_step}-layer1-{x}_")
        ]
        layer0_map[l0] = children

    saved_result[ep_id][question_id]["steps"][step_key]["frontier"] = layer0_map

    # -------- chosen_frontier (parse -> frontier paths) --------
    chosen_l0_path = None
    chosen_l1_path = None
    if os.path.exists(chosen_dir):
        chosen_candidates = sorted(
            [fn for fn in os.listdir(chosen_dir) if fn.startswith(f"{cnt_step}-frontier") and fn.endswith(".png")]
        )
        if chosen_candidates:
            last_fn = chosen_candidates[-1]
            m = re.match(rf"^{cnt_step}-frontier(\d+)_(\d+)\.png$", last_fn)
            if m:
                l0_idx, l1_idx = m.group(1), m.group(2)
                chosen_l0_path = rel_frontier_path(f"{cnt_step}-layer0-{l0_idx}.png")
                chosen_l1_path = rel_frontier_path(f"{cnt_step}-layer1-{l0_idx}_{l1_idx}.png")

    saved_result[ep_id][question_id]["steps"][step_key]["chosen_frontier"] = {
        "layer0": chosen_l0_path,
        "layer1": chosen_l1_path
    }

    # -------- memory_snapshots --------
    # memory_snapshots = {}
    # if os.path.exists(lifelong_json_path):
    #     with open(lifelong_json_path, 'r', encoding='utf-8') as f:
    #         lifelong_data = json.load(f)
    #     # lifelong_data expected: {question_id: {img_name: obj_list, ...}, ...}
    #     if question_id in lifelong_data:
    #         img2objs = lifelong_data[question_id]
    #         for img_name, obj_list in img2objs.items():
    #             if img_name.startswith(f"{cnt_step}-"): # TODO: save snapshot objects, save_snapshot_objects_with_names()
    #                 memory_snapshots[img_name] = obj_list

    # saved_result[ep_id][question_id]["steps"][step_key]["memory_snapshots"] = memory_snapshots

    # -------- final_reward (per question) --------
    q_bucket = saved_result[ep_id][question_id]
    if "final_reward" not in q_bucket:
        q_bucket["final_reward"] = "fail"
    if final_reward is not None:
        q_bucket["final_reward"] = final_reward

    # -------- write back --------
    with open(tuple_save_path, 'w', encoding='utf-8') as f:
        json.dump(saved_result, f, indent=2, ensure_ascii=False)


def _to_serializable_list(x):
    try:
        # numpy arrays / tensors
        if hasattr(x, 'tolist'):
            return x.tolist()
    except Exception:
        pass
    # tuples -> lists
    if isinstance(x, tuple):
        return list(x)
    return x


def append_step_coords_json(
    output_root_dir: str,
    question_id: str,   # include subtask_idx
    step_index: str,  # ensure consistency
    agent_position,
    agent_position_voxel,
    angle,
    target_position=None,
):
    """
    将所有问题的坐标统一增量写入实验根目录的 coords_all.json：
      {
        "<question_id>": {
          "step_0": {"agent_position": [x,y,z], "agent_position_voxel": [i,j], "angle": a, "target_position": [x,y,z] | null},
          "step_1": { ... }
        },
        ...
      }
    """
    os.makedirs(output_root_dir, exist_ok=True)
    save_path = os.path.join(output_root_dir, 'coords_all.json')

    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}

    q_bucket = data.get(question_id) or {}
    cnt_step = step_index.split('-')[-1]
    key = f"step_{cnt_step}"
    prev = q_bucket.get(key) or {}
    record = {
        "agent_position": _to_serializable_list(agent_position),
        "agent_position_voxel": _to_serializable_list(agent_position_voxel),
        "angle": float(angle) if isinstance(angle, (int, float)) or hasattr(angle, "__float__") else _to_serializable_list(angle),
        "target_position": _to_serializable_list(target_position) if target_position is not None else None,
    }
    # 若本次未提供 target_position，则保留已存在的非空 target_position，避免被覆盖为 null
    if record["target_position"] is None and isinstance(prev, dict) and prev.get("target_position") is not None:
        record["target_position"] = prev.get("target_position")
    q_bucket[key] = record
    data[question_id] = q_bucket

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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
        "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
        # "ViT-H-14", pretrained=CLIP_PATH + "/open_clip_pytorch_model.bin"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    # clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir, start_ratio, end_ratio, split, voxel_size=cfg.tsdf_grid_size
    )

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

                # mapping from the obj id in habitat to the id assigned by concept graph
                # this mapping/alignment is done by heuristic matching between object masks
                goal_obj_ids_mapping = {
                    obj_id: [] for obj_id in subtask_metadata["goal_obj_ids"]
                }

                # run steps
                task_success = False
                cnt_step = -1   # in the subtask, distinguishing from global_step especially in saving!!!
                n_filtered_snapshots = 0

                # reset tsdf planner
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                max_point_choice = None

                if cfg.clear_up_memory_every_subtask and subtask_idx > 0:
                    scene.clear_up_detections()
                    tsdf_planner = TSDFPlanner(
                        vol_bnds=tsdf_bnds,
                        voxel_size=cfg.tsdf_grid_size,
                        floor_height=floor_height,
                        floor_height_offset=0,
                        pts_init=pts,
                        init_clearance=cfg.init_clearance * 2,
                        save_visualization=cfg.save_visualization,
                    )

                while cnt_step < num_step - 1:
                    cnt_step += 1
                    global_step += 1
                    logging.info(
                        f"\n== step: {cnt_step}, global step: {global_step} =="
                    )

                    # (1) Observe the surroundings, update the scene graph and occupancy map
                    # Determine the viewing angles for the current step
                    if cnt_step == 0:
                        angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_2
                    else:
                        angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_1
                    all_angles = [
                        angle + angle_increment * (i - total_views // 2)
                        for i in range(total_views)
                    ]
                    # Let the main viewing angle be the last one to avoid potential overwriting problems
                    main_angle = all_angles.pop(total_views // 2)
                    all_angles.append(main_angle)

                    rgb_egocentric_views = []
                    all_added_obj_ids = (
                        []
                    )  # Record all the objects that are newly added in this step
                    for view_idx, ang in enumerate(all_angles):
                        # For each view
                        obs, cam_pose = scene.get_observation(pts, angle=ang)
                        rgb = obs["color_sensor"]
                        depth = obs["depth_sensor"]
                        semantic_obs = obs["semantic_sensor"]

                        # collect all view features
                        obs_file_name = f"{global_step}-view_{view_idx}.png"
                        # TODO: change to "task-{subtask_idx}_step-{cnt_step}"? 
                        with torch.no_grad():
                            # Concept graph pipeline update
                            annotated_rgb, added_obj_ids, target_obj_id_mapping = (
                                scene.update_scene_graph(
                                    image_rgb=rgb[..., :3],
                                    depth=depth,
                                    intrinsics=cam_intr,
                                    cam_pos=cam_pose,
                                    pts=pts,
                                    pts_voxel=tsdf_planner.habitat2voxel(pts),
                                    img_path=obs_file_name,
                                    frame_idx=cnt_step * total_views + view_idx,    # not used inside function
                                    ## the following in goatbench but not in aeqa
                                    semantic_obs=semantic_obs,
                                    gt_target_obj_ids=subtask_metadata["goal_obj_ids"],
                                    ## the following in aeqa but not in goatbench
                                    # target_obj_mask=None,
                                )
                            )
                            scene.all_observations[obs_file_name] = rgb ## already resized in aeqa, resize in query_vlm_for_response
                            rgb_egocentric_views.append(
                                resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                            )
                            if cfg.save_visualization:
                                plt.imsave(
                                    os.path.join(eps_snapshot_dir, obs_file_name),
                                    annotated_rgb,
                                )
                            else:
                                plt.imsave(
                                    os.path.join(eps_snapshot_dir, obs_file_name), rgb
                                )
                            # update the mapping of hm3d object id to our detected object id
                            ## goal_obj_ids_mapping not in aeqa
                            for (
                                gt_goal_id,
                                det_goal_id,
                            ) in target_obj_id_mapping.items():
                                goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
                            all_added_obj_ids += added_obj_ids

                        # Clean up or merge redundant objects periodically
                        scene.periodic_cleanup_objects(
                            frame_idx=cnt_step * total_views + view_idx,    # for processing_needed()
                            pts=pts,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,  ## not in aeqa
                        )

                        # Update depth map, occupancy map
                        tsdf_planner.integrate(
                            color_im=rgb,
                            depth_im=depth,
                            cam_intr=cam_intr,
                            cam_pose=pose_habitat_to_tsdf(cam_pose),
                            obs_weight=1.0,
                            margin_h=int(cfg.margin_h_ratio * img_height),
                            margin_w=int(cfg.margin_w_ratio * img_width),
                            explored_depth=cfg.explored_depth,
                        )
                    logging.info(f"Goal object mapping: {goal_obj_ids_mapping}")

                    # (2) Update Memory Snapshots with hierarchical clustering
                    # Choose all the newly added objects as well as the objects nearby as the cluster targets
                    all_added_obj_ids = [
                        obj_id
                        for obj_id in all_added_obj_ids
                        if obj_id in scene.objects  # list(scene.objects.values())[0].keys()
                    ]
                    for obj_id, obj in scene.objects.items():
                        if (
                            np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                            < cfg.scene_graph.obj_include_dist + 0.5
                        ):
                            all_added_obj_ids.append(obj_id)
                    scene.update_snapshots(
                        obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
                    )   # ['all_observations', 'frames', 'snapshots', ...]
                    logging.info(
                        f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
                    )

                    # (3) Update the Frontier Snapshots
                    update_success = tsdf_planner.update_frontier_map(
                        pts=pts,
                        cfg=cfg.planner,
                        scene=scene,
                        # cnt_step=cnt_step,  # for saving frontier image
                        cnt_step=f"task-{subtask_idx}_step-{cnt_step}",
                        save_frontier_image=cfg.save_visualization,
                        eps_frontier_dir=eps_frontier_dir,  # for episode
                        prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
                    )
                    if not update_success:
                        logging.info("Warning! Update frontier map failed!")
                        ## in aeqa, if the first step fails, we should stop, change question_id to subtask_id?
                        # if cnt_step == 0:
                        #     logging.info(
                        #         f"subtask id {subtask_id} invalid: update_frontier_map failed!"
                        #     )
                        #     break

                    # (4) Choose the next navigation point by querying the VLM
                    if cfg.choose_every_step:
                        # if we choose to query vlm every step, we clear the target point every step
                        if (
                            tsdf_planner.max_point is not None
                            and type(tsdf_planner.max_point) == Frontier
                        ):
                            # reset target point to allow the model to choose again
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None

                    # use the most common id in the mapped ids as the detected target object id
                    target_obj_ids_estimate = []
                    for obj_id, det_ids in goal_obj_ids_mapping.items():
                        if len(det_ids) == 0:
                            continue
                        target_obj_ids_estimate.append(
                            max(set(det_ids), key=det_ids.count)
                        )
                        
                    ## add chosen frontier dir
                    chosen_frontier_path = os.path.join(episode_dir, 'chosen_frontier')

                    if (
                        tsdf_planner.max_point is None
                        and tsdf_planner.target_point is None
                    ):
                        ## ensure replay_step_info.json exists before VLM query (for context recall)
                        tuple_save_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, 'replay_step_info.json')
                        try:
                            tuple_step_save(
                                tuple_save_path=tuple_save_path,
                                question_id=subtask_metadata["question_id"],
                                question=subtask_metadata["question"],
                                cnt_step=f"task-{subtask_idx}_step-{cnt_step}",
                                cfg=cfg,
                                lifelong_json_path=lifelong_json_path,
                                subtask_metadata=subtask_metadata,
                            )
                        except Exception as e:
                            logging.info(f"[ReplaySim] Pre-create replay json failed: {e}")
                        
                        # query the VLM for the next navigation point, and the reason for the choice
                        vlm_response = query_vlm_for_response(
                            subtask_metadata=subtask_metadata,
                            scene=scene,
                            tsdf_planner=tsdf_planner,
                            rgb_egocentric_views=rgb_egocentric_views,
                            cfg=cfg,
                            verbose=True,
                            ##
                            chosen_frontier_path=chosen_frontier_path,  # load and save chosen_frontier image, tuple_step_save()
                            step_idx=f"task-{subtask_idx}_step-{cnt_step}", # saving frontier
                            # question_id=question_id,
                            lifelong_json_path=lifelong_json_path,
                        )
                        if vlm_response is None:
                            logging.info(
                                f"Subtask id {subtask_id} invalid: query_vlm_for_response failed!"
                            )
                            break

                        # max_point_choice, n_filtered_snapshots = vlm_response
                        max_point_choice, gpt_answer, n_filtered_snapshots = vlm_response   # logging.info(gpt_answer)
                        
                        ##
                        tuple_save_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, 'replay_step_info.json')
                        tuple_step_save(
                            tuple_save_path=tuple_save_path,
                            question_id=subtask_metadata["question_id"],
                            question=subtask_metadata["question"],
                            cnt_step=f"task-{subtask_idx}_step-{cnt_step}",
                            cfg=cfg,
                            lifelong_json_path=lifelong_json_path,
                            subtask_metadata=subtask_metadata,
                        )

                        # set the vlm choice as the navigation target
                        update_success = tsdf_planner.set_next_navigation_point(
                            choice=max_point_choice,
                            pts=pts,
                            objects=scene.objects,
                            cfg=cfg.planner,
                            pathfinder=scene.pathfinder,
                        )
                        if not update_success:
                            logging.info(
                                f"Subtask id {subtask_id} invalid: set_next_navigation_point failed!"
                            )
                            break
                        
                        ## 
                        try:
                            target_pos = None
                            try:
                                target_pos = getattr(max_point_choice, 'position', None)
                            except Exception:
                                target_pos = None
                            append_step_coords_json(
                                output_root_dir=cfg.output_dir,
                                question_id=subtask_metadata["question_id"],
                                step_index=f"task-{subtask_idx}_step-{cnt_step}",
                                agent_position=pts,
                                agent_position_voxel=tsdf_planner.habitat2voxel(pts)[:2],
                                angle=angle,
                                target_position=target_pos,
                            )
                        except Exception as e:
                            logging.info(f"[Coords] Failed to append target position: {e}")

                    # (5) Agent navigate to the target point for one step
                    return_values = tsdf_planner.agent_step(
                        pts=pts,
                        angle=angle,
                        objects=scene.objects,
                        snapshots=scene.snapshots,
                        pathfinder=scene.pathfinder,
                        cfg=cfg.planner,
                        path_points=None,
                        save_visualization=cfg.save_visualization,
                    )   # TODO: UserWarning: *c* argument looks like a single numeric RGB or RGBA sequence
                    if return_values[0] is None:
                        logging.info(
                            f"Subtask id {subtask_id} invalid: agent_step failed!"
                        )
                        break

                    # update agent's position and rotation
                    pts, angle, pts_voxel, fig, _, target_arrived = return_values
                    logger.log_step(pts_voxel=pts_voxel)
                    logging.info(
                        f"Current position: {pts}, {logger.subtask_explore_dist:.3f}"
                    )
                    
                    ##
                    try:
                        append_step_coords_json(
                            output_root_dir=cfg.output_dir,
                            question_id=subtask_metadata["question_id"],
                            step_index=f"task-{subtask_idx}_step-{cnt_step}",
                            agent_position=pts,
                            agent_position_voxel=pts_voxel[:2] if hasattr(pts_voxel, '__len__') else pts_voxel,
                            angle=angle,
                            target_position=None,
                        )
                    except Exception as e:
                        logging.info(f"[Coords] Failed to append agent pose after step: {e}")

                    # sanity check about objects, scene graph, snapshots, ...
                    scene.sanity_check(cfg=cfg)

                    if cfg.save_visualization:
                        # save the top-down visualization
                        logger.save_topdown_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            subtask_metadata=subtask_metadata,
                            goal_obj_ids_mapping=goal_obj_ids_mapping,
                            fig=fig,
                        )
                        # save the visualization of vlm's choice at each step
                        logger.save_frontier_visualization(
                            global_step=global_step,
                            subtask_id=subtask_id,
                            tsdf_planner=tsdf_planner,
                            max_point_choice=max_point_choice,
                            global_caption=f"{subtask_metadata['question']}\n{subtask_metadata['task_type']}\n{subtask_metadata['class']}",
                        )   ## as an example to adapt aeqa setting to goat-bench

                    # (6) Check if the agent has arrived at the target to finish the question
                    if type(max_point_choice) == SnapShot and target_arrived:
                        
                        # 
                        tuple_step_save(
                            tuple_save_path=tuple_save_path,
                            question_id=subtask_metadata["question_id"],
                            question=subtask_metadata["question"],
                            cnt_step=f"task-{subtask_idx}_step-{cnt_step}",
                            cfg=cfg,
                            lifelong_json_path=lifelong_json_path,
                            subtask_metadata=subtask_metadata,
                            final_reward="pass"
                        )
                        
                        # when the target is a snapshot, and the agent arrives at the target
                        # we consider the subtask is finished, take an observation and save the chosen target snapshot
                        obs, _ = scene.get_observation(pts, angle=angle)
                        rgb = obs["color_sensor"]
                        plt.imsave(
                            os.path.join(
                                logger.subtask_object_observe_dir, f"target.png"
                            ),
                            rgb,
                        )

                        snapshot_filename = max_point_choice.image.split(".")[0]
                        os.system(
                            f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(logger.subtask_object_observe_dir, f'snapshot_{snapshot_filename}.png')}"
                        )

                        task_success = True
                        break

                # get some statistics
                if task_success and np.any(
                    [
                        obj_id in max_point_choice.cluster
                        for obj_id in target_obj_ids_estimate
                    ]
                ):
                    success_by_snapshot = True
                    logging.info(
                        f"Success: {target_obj_ids_estimate} in chosen snapshot {max_point_choice.image}!"
                    )
                else:
                    success_by_snapshot = False
                    logging.info(
                        f"Fail: {target_obj_ids_estimate} not in chosen snapshot!"
                    )
                # calculate the distance to the nearest view point
                agent_subtask_distance = calc_agent_subtask_distance(
                    pts, subtask_metadata["viewpoints"], scene.pathfinder
                )
                if agent_subtask_distance < cfg.success_distance:
                    success_by_distance = True
                    logging.info(
                        f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!"
                    )
                else:
                    success_by_distance = False
                    logging.info(
                        f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!"
                    )

                logger.log_subtask_result(
                    success_by_snapshot=success_by_snapshot,
                    success_by_distance=success_by_distance,
                    subtask_id=subtask_id,
                    gt_subtask_explore_dist=subtask_metadata["gt_subtask_explore_dist"],
                    goal_type=goal_type,
                    n_filtered_snapshots=n_filtered_snapshots,
                    n_total_snapshots=len(scene.snapshots),
                    n_total_frames=len(scene.frames),
                )

                logging.info(f"Scene graph of question {subtask_id}:")
                logging.info(f"Question: {subtask_metadata['question']}")
                logging.info(f"Task type: {subtask_metadata['task_type']}")
                logging.info(f"Answer: {subtask_metadata['class']}")
                scene.print_scene_graph()

                if not cfg.save_visualization:
                    # clear up the stored images to save memory
                    os.system(
                        f"rm -r {os.path.join(str(cfg.output_dir), f'{subtask_id}')}"
                    )

            # save the results at the end of each episode
            logger.save_results()

            logging.info(f"Episode {episode_id} finish")
            if not cfg.save_visualization:
                os.system(f"rm -r {episode_dir}")

    logger.save_results()
    # aggregate the results from different splits into a single file
    logger.aggregate_results()

    logging.info(f"All scenes finish")


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
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)
    # CLI overrides for replay recall behavior
    cfg.replay_mode = args.replay_mode
    cfg.replay_top = args.replay_top
    if args.retrieve_root:
        cfg.retrieve_root = args.retrieve_root

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