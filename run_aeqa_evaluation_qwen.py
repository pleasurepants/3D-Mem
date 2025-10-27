import os, json, re
from typing import Optional, Dict, List
import uuid
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"


import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import time
from typing import Optional
import logging
from src.const import *
from sentence_transformers import SentenceTransformer, util
import numpy as np
client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)
SENTENCE_TRANSFORMERS_PATH = '/nfs/data8/jingpei/eqa/models/clip-ViT-B-32'
CLIP_PATH = '/nfs/data8/jingpei/eqa/models/CLIP-ViT-H-14-laion2B-s32B-b79K'




import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner_hdbscan import TSDFPlanner, Frontier, SnapShot
from src.scene_aeqa import Scene
from src.utils import resize_image, get_pts_angle_aeqa
from src.query_vlm_aeqa_qwen import query_vlm_for_response
from src.logger_aeqa import Logger
from src.const import *


def question_room(folder_path, questions_list_path):
    # 1. 提取question_id
    question_id = os.path.basename(folder_path.rstrip('/'))

    # 2. 读取json
    with open(questions_list_path, 'r') as f:
        questions = json.load(f)

    # 3. 查找question_id对应的episode_history
    episode_history = None
    for q in questions:
        if q.get('question_id') == question_id:
            episode_history = q.get('episode_history')
            break

    # 4. 统计所有episode_history对应的question_id
    ep_history_to_question_ids = {}
    for q in questions:
        ep_history = q.get('episode_history')
        if ep_history not in ep_history_to_question_ids:
            ep_history_to_question_ids[ep_history] = []
        ep_history_to_question_ids[ep_history].append(q.get('question_id'))

    # 返回该question_id对应的episode_history，以及这个episode_history下的所有question_id列表
    if episode_history is not None:
        return episode_history, ep_history_to_question_ids[episode_history]
    else:
        return None, []



def check_lifelong_memory(lifelong_json_path, lifelong_memory, cfg, question):
    """
    用于检索所有历史memory snapshot能否直接回答问题。
    - 若API能回答，返回 full_response（如"Snapshot 0 XXX"）
    - 若所有memory都不能回答，返回 "Snapshot -1 No Snapshot is available"
    """

    def format_content(contents):
        formated_content = []
        for c in contents:
            formated_content.append({"type": "text", "text": c[0]})
            if len(c) == 2:
                formated_content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{c[1]}",
                            "detail": "high",
                        },
                    }
                )
        return formated_content

    def call_openai_api(sys_prompt, contents) -> Optional[str]:
        max_tries = 5
        retry_count = 0
        formated_content = format_content(contents)
        message_text = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": formated_content},
        ]
        while retry_count < max_tries:
            try:
                completion = client.chat.completions.create(
                    model="qwen",  # gpt-4o-qwen-minicpm-qwen
                    messages=message_text,
                    temperature=0.7,
                    max_tokens=4096, # 4096 for gpt-4o
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                return completion.choices[0].message.content
            except openai.RateLimitError as e:
                print("Rate limit error, waiting for 60s")
                time.sleep(30)
                retry_count += 1
                continue
            except Exception as e:
                print("Error: ", e)
                time.sleep(60)
                retry_count += 1
                continue

        return None



    def build_lifelong_context_prompt(
        question,
        img_info_list,    # List[(img_path, obj_list)]
        cfg
    ):
        sys_prompt = ""
        sys_prompt += "Task: You are an agent exploring an indoor environment to answer a specific question.\n"
        sys_prompt += "You are given several past observations, each with an image and a list of detected objects.\n"
        sys_prompt += "Your goal is to combine all the provided information and reason about the best next action for exploration, or whether any area already contains enough information to answer the question.\n"
        sys_prompt += "You may suggest to focus on an area that already looks promising, or point out what is still missing and where to explore next.\n"
        sys_prompt += "Do NOT refer to the images by their order or index (do not say 'first/second/third image').\n"
        sys_prompt += "Follow a step-by-step reasoning (chain-of-thought) process as described below, but your final output should be a single, coherent paragraph summarizing your suggestion.\n"
        sys_prompt += "\n"
        sys_prompt += "Chain-of-thought reasoning steps:\n"
        sys_prompt += "Step 1: Review all object lists observed so far. Consider what these clues reveal about the environment.\n"
        sys_prompt += "Step 2: Reflect on the question and determine whether any area already looks promising for answering the question, or what is still missing (key clues, objects, or room types).\n"
        sys_prompt += "Step 3: Synthesize your reasoning to suggest the most effective next action—this may be to further explore certain areas, or to focus on a specific location that already looks suitable for answering the question. Clearly explain your reasoning, combining all observations.\n"
        sys_prompt += "DO NOT refer to images by order or index. Your final output must be a single, well-formed paragraph targeting the question, providing clear and actionable guidance for what to do next.\n"

        content = []
        content += [(f"Question: {question}",)]
        content += [("Here are your current memory observations (object lists):",)]

        for i, (img_path, obj_list, _) in enumerate(img_info_list):
            abs_img_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, img_path)
            if not os.path.exists(abs_img_path):
                print(f"Warning: image file {abs_img_path} does not exist!")
                continue
            with open(abs_img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content += [(f"Snapshot {i+1}", img_base64)]
            text = ", ".join(obj_list)
            content += [(text,)]
            content += [(" ",)]

        content += [(
            "Please use the above object lists and reasoning steps, but only output your final conclusion as a single paragraph. "
            "You must NOT refer to any image or observation by order, number, or index (do not say 'first', 'second', 'third', or similar words). "
            "Your answer should always be forward-looking and actionable, based on the combined clues: "
            "If any area or observed set of objects appears promising for answering the question, you may recommend focusing future exploration or further verification on that area. "
            "If none of the current areas or object combinations seem relevant, you should suggest which type of area, room, or object may be worth exploring or searching for. "
            "Do NOT declare the task complete or say that no further action is needed. "
            "Your answer must give a clear direction or focus for exploration—such as which area to prioritize, or what room or object to look for—without implying a step-by-step plan."
        ,)]


        return sys_prompt, content

    def select_topk_snapshots_by_clip(
        lifelong_json_path,
        available_history,
        question,
        top_k=3,
        max_objects=8,
        # model_name='/anvme/workspace/v100dd12-3dmem/model/clip-ViT-B-32'
        # model_name='sentence-transformers/clip-ViT-B-32'  # https://huggingface.co/sentence-transformers/clip-ViT-B-32
        model_name=SENTENCE_TRANSFORMERS_PATH
    ):

        def obj_list_to_caption(obj_list):
            """
            将object list转成简洁的英文场景描述，用于CLIP文本embedding。
            """
            if not obj_list:
                return "This image does not contain any recognizable objects."
            # 你可以扩展更复杂的模板（比如按物品类别判断房间类型）
            return "This image contains: " + ", ".join(obj_list) + "."
        """
        输入:
            lifelong_json_path: str, json路径
            available_history: list, 要筛选的question_id列表
            question: str, 当前问题
            top_k: int, 选出top-k
            max_objects: int, 每张图片保留最多几个物品
            model_name: str, sentence-transformers模型路径或名称
        输出:
            List[Tuple[snapshot_path, obj_list]]
            snapshot_path: "question_id/snapshot/img.png"
        """
        # 加载json
        with open(lifelong_json_path, 'r') as f:
            lifelong_json = json.load(f)

        all_img_keys = []
        all_obj_lists = []
        all_obj_texts = []

        for qid in available_history:
            if qid not in lifelong_json:
                continue
            img2objs = lifelong_json[qid]
            for img_name, obj_list in img2objs.items():
                obj_list_trunc = obj_list[:max_objects]
                # 用自然语言模板描述物品
                obj_text = obj_list_to_caption(obj_list_trunc)
                # 路径按你的格式拼接
                snapshot_path = f"{qid}/snapshot/{img_name}"
                all_img_keys.append(snapshot_path)
                all_obj_lists.append(obj_list_trunc)
                all_obj_texts.append(obj_text)

        if len(all_img_keys) == 0:
            print("No available snapshots!")
            return []

        # 做embedding
        model = SentenceTransformer(model_name)
        snapshot_embs = model.encode(all_obj_texts, convert_to_tensor=True)
        question_emb = model.encode([question], convert_to_tensor=True)
        sims = util.cos_sim(question_emb, snapshot_embs)[0]  # shape: (n_snapshots,)

        # 取top-k
        topk_idx = np.array(sims.cpu()).argsort()[-top_k:][::-1]
        results = []
        for i in topk_idx:
            results.append((all_img_keys[i], all_obj_lists[i], all_obj_texts[i]))
            logging.info((f"{all_img_keys[i]}  -->  {all_obj_lists[i]}"))
        return results






    # 1. 取当前question_id
    with open(cfg.questions_list_path, 'r') as f:
        import json
        questions_list = json.load(f)
    question_id = None
    for q in questions_list:
        if q["question"] == question:
            question_id = q["question_id"]
            break
    if question_id is None:
        raise ValueError(f"Question not found: {question}")

    # 2. 读取lifelong_json
    os.makedirs(os.path.dirname(lifelong_json_path), exist_ok=True)

    if not os.path.exists(lifelong_json_path):
        # 如果文件不存在，直接返回没有可用的历史记忆
        return "Snapshot -1 No Snapshot is available(No lifelong memory available)"
    
    with open(lifelong_json_path, 'r') as f:
        lifelong_json = json.load(f)

    # ---- 这里检查有没有真正可用的历史记忆 ----
    available_history = [
        other_qid for other_qid in lifelong_memory
        if other_qid != question_id and other_qid in lifelong_json
    ]
    if len(available_history) == 0:
        return "Snapshot -1 No Snapshot is available(No lifelong memory available)"

    
    # 3. 获取top-k相关的snapshot图片路径和物品列表
    top3 = select_topk_snapshots_by_clip(
        lifelong_json_path, available_history, question, top_k=3
    )

    sys_prompt, content = build_lifelong_context_prompt(question, top3, cfg)
    full_response = call_openai_api(sys_prompt, content)

    if full_response is not None:
        return full_response

    return "Snapshot -1 No Snapshot is available"




def tuple_step_save(
    tuple_save_path: str,
    question_id: str,
    question: str,
    cnt_step: int,
    cfg,
    lifelong_json_path: str,
    question_data: Optional[dict] = None,   # may carry {"episode_history": "..."}
    episode_history_id: Optional[str] = None,  # explicit episode id, if known
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

    # priority: existing > explicit param > question_data > auto-generate
    ep_id = (
        existing_ep
        or episode_history_id
        or (question_data.get("episode_history") if question_data else None)
        or f"episode-{uuid.uuid4().hex[:8]}"
    )

    if ep_id not in saved_result:
        saved_result[ep_id] = {}

    # init question bucket under this episode
    if question_id not in saved_result[ep_id]:
        saved_result[ep_id][question_id] = {"question": question, "steps": {}}
    else:
        # keep existing question text if already stored; otherwise set it
        saved_result[ep_id][question_id].setdefault("question", question)
        saved_result[ep_id][question_id].setdefault("steps", {})

    step_key = f"step_{cnt_step}"
    saved_result[ep_id][question_id]["steps"][step_key] = {}

    # --- dirs ---
    q_root = os.path.join(cfg.output_parent_dir, cfg.exp_name, question_id)
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
    memory_snapshots = {}
    if os.path.exists(lifelong_json_path):
        with open(lifelong_json_path, 'r', encoding='utf-8') as f:
            lifelong_data = json.load(f)
        # lifelong_data expected: {question_id: {img_name: obj_list, ...}, ...}
        if question_id in lifelong_data:
            img2objs = lifelong_data[question_id]
            for img_name, obj_list in img2objs.items():
                if img_name.startswith(f"{cnt_step}-"):
                    memory_snapshots[img_name] = obj_list

    saved_result[ep_id][question_id]["steps"][step_key]["memory_snapshots"] = memory_snapshots

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
    question_id: str,
    step_index: int,
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
    key = f"step_{step_index}"
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

def main(cfg, start_ratio=0.0, end_ratio=1.0):
    logging.info("[DEBUG] Starting main function...")
    # load the default concept graph config
    logging.info("[DEBUG] Loading concept graph config...")
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)
    logging.info("[DEBUG] Concept graph config loaded successfully")

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    logging.info("[DEBUG] Loading questions dataset...")
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    questions_list = sorted(questions_list, key=lambda x: x["question_id"])
    logging.info(f"Total number of questions: {total_questions}")
    # only process a subset of the questions
    questions_list = questions_list[
        int(start_ratio * total_questions) : int(end_ratio * total_questions)
    ]
    logging.info(f"number of questions after splitting: {len(questions_list)}")
    logging.info(f"question path: {cfg.questions_list_path}")
    logging.info("[DEBUG] Questions dataset loaded successfully")

    # load detection and segmentation models
    logging.info("[DEBUG] Loading YOLO model...")
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    logging.info("[DEBUG] Loading SAM model...")
    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    logging.info("[DEBUG] Loading CLIP model...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        # "ViT-H-14", pretrained="/anvme/workspace/v100dd12-3dmem/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"  # "ViT-H-14", "laion2b_s32b_b79k"
        "ViT-H-14", pretrained=CLIP_PATH + "/open_clip_pytorch_model.bin"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")
    logging.info("[DEBUG] All models loaded successfully")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir,
        start_ratio,
        end_ratio,
        len(questions_list),
        voxel_size=cfg.tsdf_grid_size,
    )

    # Run all questions
    logging.info(f"[DEBUG] Starting to process {len(questions_list)} questions...")
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data["question_id"]
        scene_id = question_data["episode_history"]
        if question_id in logger.success_list or question_id in logger.fail_list:
            logging.info(f"Question {question_id} already processed")
            continue
        if any([invalid_scene_id in scene_id for invalid_scene_id in INVALID_SCENE_ID]):
            logging.info(f"Skip invalid scene {scene_id}")
            continue
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        question = question_data["question"]
        answer = question_data["answer"]
        pts, angle = get_pts_angle_aeqa(
            question_data["position"], question_data["rotation"]
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
        tsdf_planner = TSDFPlanner(
            vol_bnds=get_scene_bnds(scene.pathfinder, floor_height=pts[1])[0],
            voxel_size=cfg.tsdf_grid_size,
            floor_height=pts[1],
            floor_height_offset=0,
            pts_init=pts,
            init_clearance=cfg.init_clearance * 2,
            save_visualization=cfg.save_visualization,
        )

        episode_dir, eps_chosen_snapshot_dir, eps_frontier_dir, eps_snapshot_dir = (
            logger.init_episode(
                question_id=question_id,
                init_pts_voxel=tsdf_planner.habitat2voxel(pts)[:2],
            )
        )

        logging.info(f"\n\nQuestion id {question_id} initialization successful!")

        # run steps
        task_success = False
        cnt_step = -1


        # starting lifelong-memory
        lifelong_memory = question_room(episode_dir, cfg.questions_list_path)
        lifelong_json_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, "lifelong_storage.json")
        lifelong_memory = lifelong_memory[1]
        # lifelong_context = check_lifelong_memory(lifelong_json_path, lifelong_memory, cfg, question)



        gpt_answer = None
        n_filtered_snapshots = 0
        while cnt_step < cfg.num_step - 1:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")

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
                obs, cam_pose = scene.get_observation(pts, ang)
                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]

                obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                with torch.no_grad():
                    # Concept graph pipeline update
                    annotated_rgb, added_obj_ids, _ = scene.update_scene_graph(
                        image_rgb=rgb[..., :3],
                        depth=depth,
                        intrinsics=cam_intr,
                        cam_pos=cam_pose,
                        pts=pts,
                        pts_voxel=tsdf_planner.habitat2voxel(pts),
                        img_path=obs_file_name,
                        frame_idx=cnt_step * total_views + view_idx,
                        target_obj_mask=None,
                    )
                    resized_rgb = resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                    scene.all_observations[obs_file_name] = resized_rgb
                    rgb_egocentric_views.append(resized_rgb)
                    if cfg.save_visualization:
                        plt.imsave(
                            os.path.join(eps_snapshot_dir, obs_file_name), annotated_rgb
                        )
                    else:
                        plt.imsave(os.path.join(eps_snapshot_dir, obs_file_name), rgb)
                    all_added_obj_ids += added_obj_ids

                # Clean up or merge redundant objects periodically
                scene.periodic_cleanup_objects(
                    frame_idx=cnt_step * total_views + view_idx, pts=pts
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

            # (2) Update Memory Snapshots with hierarchical clustering
            # Choose all the newly added objects as well as the objects nearby as the cluster targets
            all_added_obj_ids = [
                obj_id for obj_id in all_added_obj_ids if obj_id in scene.objects
            ]
            for obj_id, obj in scene.objects.items():
                if (
                    np.linalg.norm(obj["bbox"].center[[0, 2]] - pts[[0, 2]])
                    < cfg.scene_graph.obj_include_dist + 0.5
                ):
                    all_added_obj_ids.append(obj_id)
            scene.update_snapshots(
                obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection
            )
            logging.info(
                f"Step {cnt_step}, update snapshots, {len(scene.objects)} objects, {len(scene.snapshots)} snapshots"
            )

            # (3) Update the Frontier Snapshots
            update_success = tsdf_planner.update_frontier_map(
                pts=pts,
                cfg=cfg.planner,
                scene=scene,
                cnt_step=cnt_step,
                save_frontier_image=cfg.save_visualization,
                eps_frontier_dir=eps_frontier_dir,
                prompt_img_size=(cfg.prompt_h, cfg.prompt_w),
            )
            if not update_success:
                logging.info("Warning! Update frontier map failed!")
                if cnt_step == 0:  # if the first step fails, we should stop
                    logging.info(
                        f"Question id {question_id} invalid: update_frontier_map failed!"
                    )
                    break

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


            # add chosen frontier dir
            chosen_frontier_path = os.path.join(episode_dir, 'chosen_frontier')


            


            if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                # ensure replay_step_info.json exists before VLM query (for context recall)
                tuple_save_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, 'replay_step_info.json')
                try:
                    tuple_step_save(
                        tuple_save_path=tuple_save_path,
                        question_id=question_id,
                        question=question,
                        cnt_step=cnt_step,
                        cfg=cfg,
                        lifelong_json_path=lifelong_json_path,
                        question_data=question_data,
                    )
                except Exception as e:
                    logging.info(f"[ReplaySim] Pre-create replay json failed: {e}")

                # query the VLM for the next navigation point, and the reason for the choice
                # annotate identifiers into cfg for prompt construction
                try:
                    cfg.episode_history_id = scene_id
                    cfg.current_question_id = question_id
                    cfg.current_question_text = question
                except Exception:
                    pass
                vlm_response = query_vlm_for_response(
                    question=question,
                    scene=scene,
                    tsdf_planner=tsdf_planner,
                    rgb_egocentric_views=rgb_egocentric_views,
                    cfg=cfg,
                    verbose=True,
                    chosen_frontier_path=chosen_frontier_path,
                    step_idx=cnt_step,
                    question_id=question_id,
                    lifelong_json_path=lifelong_json_path,
                    exp_tuple_path=(args.exp_tuple if isinstance(args.exp_tuple, str) and len(args.exp_tuple) > 0 else None),
                    inject_experience=(False if str(getattr(cfg, 'replay_mode', 'sim')).startswith('traj') else bool(args.caption)),
                    inject_critique=(False if str(getattr(cfg, 'replay_mode', 'sim')).startswith('traj') else bool(args.critique)),
                    inject_abstraction=(True if str(getattr(cfg, 'replay_mode', 'sim')).startswith('traj') else bool(args.abstraction)),
                    # lifelong_context=lifelong_context,
                )
                if vlm_response is None:
                    logging.info(
                        f"Question id {question_id} invalid: query_vlm_for_response failed!"
                    )
                    break
                
                
                max_point_choice, gpt_answer, n_filtered_snapshots = vlm_response


                # tuple save
                tuple_save_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, 'replay_step_info.json')
                tuple_step_save(
                    tuple_save_path=tuple_save_path,
                    question_id=question_id,
                    question=question,
                    cnt_step=cnt_step,
                    cfg=cfg,
                    lifelong_json_path=lifelong_json_path,
                    question_data=question_data,
                )




                # set the vlm choice as the navigation target
                update_success = tsdf_planner.set_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts,
                    objects=scene.objects,
                    cfg=cfg.planner,
                    pathfinder=scene.pathfinder,
                    random_position=False,
                )
                if not update_success:
                    logging.info(
                        f"Question id {question_id} invalid: set_next_navigation_point failed!"
                    )
                    break

                # 记录选择后的目标坐标（frontier 的世界坐标）到 coords_all.json（统一文件）
                try:
                    target_pos = None
                    try:
                        # Frontier 类型有 position 属性
                        target_pos = getattr(max_point_choice, 'position', None)
                    except Exception:
                        target_pos = None
                    append_step_coords_json(
                        output_root_dir=cfg.output_dir,
                        question_id=question_id,
                        step_index=cnt_step,
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
            )
            if return_values[0] is None:
                logging.info(f"Question id {question_id} invalid: agent_step failed!")
                break

            # update agent's position and rotation
            pts, angle, pts_voxel, fig, _, target_arrived = return_values
            logger.log_step(pts_voxel=pts_voxel)
            logging.info(f"Current position: {pts}, {logger.explore_dist:.3f}")

            # 追加写入当前step后的 agent 位姿（无 target 变化）到 coords_all.json（统一文件）
            try:
                append_step_coords_json(
                    output_root_dir=cfg.output_dir,
                    question_id=question_id,
                    step_index=cnt_step,
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
                    cnt_step=cnt_step,
                    fig=fig,
                )
                # save the visualization of vlm's choice at each step
                logger.save_frontier_visualization(
                    cnt_step=cnt_step,
                    tsdf_planner=tsdf_planner,
                    max_point_choice=max_point_choice,
                    global_caption=f"{question}\n{answer}",
                )

            # (6) Check if the agent has arrived at the target to finish the question
            if type(max_point_choice) == SnapShot and target_arrived:


                #set final reward as succcess
                tuple_step_save(
                    tuple_save_path=tuple_save_path,
                    question_id=question_id,
                    question=question,
                    cnt_step=cnt_step,
                    cfg=cfg,
                    lifelong_json_path=lifelong_json_path,
                    question_data=question_data,
                    final_reward="pass"
                )

                # when the target is a snapshot, and the agent arrives at the target
                # we consider the question is finished and save the chosen target snapshot
                snapshot_filename = max_point_choice.image.split(".")[0]
                os.system(
                    f"cp {os.path.join(eps_snapshot_dir, max_point_choice.image)} {os.path.join(eps_chosen_snapshot_dir, f'snapshot_{snapshot_filename}.png')}"
                )

                task_success = True
                logging.info(
                    f"Question id {question_id} finished after arriving at target!"
                )
                break

        logger.log_episode_result(
            success=task_success,
            question_id=question_id,
            explore_dist=logger.explore_dist,
            gpt_answer=gpt_answer,
            n_filtered_snapshots=n_filtered_snapshots,
            n_total_snapshots=len(scene.snapshots),
            n_total_frames=len(scene.frames),
        )

        logging.info(f"Scene graph of question {question_id}:")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")
        logging.info(f"Prediction: {gpt_answer}")
        scene.print_scene_graph()

        # update the saved results after each episode
        logger.save_results()

        if not cfg.save_visualization:
            # clear up the stored images to save memory
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
    parser.add_argument("--replay_mode", help="replay selection mode: sim or random or traj_sim or traj_random", default="sim", type=str)
    parser.add_argument("--replay_top", help="top-k for replay candidates", default=1, type=int)
    parser.add_argument("--retrieve_root", help="external retrieve root; expects replay_step_info.json & experience_output.json inside", default="", type=str)
    parser.add_argument("--use_episodic_context", help="whether to enable episodic context (0/1)", default=1, type=int)
    parser.add_argument("--chat_seed", help="random seed for vLLM generation (decoupled from cfg.seed)", default=None, type=int)
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
    # CLI overrides for replay recall behavior
    cfg.replay_mode = str(args.replay_mode).strip().lower()
    cfg.replay_top = args.replay_top
    if args.retrieve_root:
        cfg.retrieve_root = args.retrieve_root
    # Episodic context toggle
    cfg.use_episodic_context = bool(args.use_episodic_context)
    # vLLM per-request seed (independent from cfg.seed)
    if args.chat_seed is not None:
        cfg.chat_seed = int(args.chat_seed)
    # traj_* external file path
    if args.traj_file:
        cfg.traj_file = args.traj_file
    
    # ppl_rank parameters
    if args.ppl_rank:
        cfg.ppl_rank = str(args.ppl_rank).strip().lower()
        cfg.ppl_rank_file = args.ppl_rank_file

    # normalize inject_stage into cfg (empty -> None)
    # normalize stage flag (exp_at preferred)
    try:
        _stage = str(args.exp_at).strip().lower()
        if _stage in ("bvf", "cvf"):
            cfg.exp_at = _stage
        else:
            cfg.exp_at = None
    except Exception:
        cfg.exp_at = None
    # backward compatibility mirrors
    try:
        cfg.replay_at = cfg.exp_at
    except Exception:
        pass
    try:
        cfg.inject_stage = cfg.exp_at
    except Exception:
        pass

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(
        str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log"
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

    # If chat_seed is provided, propagate to environment so all API calls (even without explicit seed) use it
    try:
        if hasattr(cfg, 'chat_seed') and cfg.chat_seed is not None:
            os.environ['VLLM_SEED'] = str(int(cfg.chat_seed))
    except Exception:
        pass

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    try:
        logging.info(
            f"[ChatSeed] chat_seed={getattr(cfg, 'chat_seed', None)} | VLLM_SEED={os.getenv('VLLM_SEED')}"
        )
    except Exception:
        pass
    main(cfg, args.start_ratio, args.end_ratio)
