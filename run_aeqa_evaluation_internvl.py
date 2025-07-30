import os

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
import re
import json
from sentence_transformers import SentenceTransformer, util
import numpy as np
client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)




import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt

import open_clip
from ultralytics import SAM, YOLOWorld

from src.habitat import pose_habitat_to_tsdf
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_planner_hdbscan import TSDFPlanner, Frontier, SnapShot
from src.scene_aeqa import Scene
from src.utils import resize_image, get_pts_angle_aeqa
from src.query_vlm_aeqa_internvl import query_vlm_for_response
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
                    model="internvl",  # gpt-4o-internvl-minicpm-qwen
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
        model_name='/anvme/workspace/v100dd12-3dmem/model/clip-ViT-B-32'
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






def main(cfg, start_ratio=0.0, end_ratio=1.0):
    # load the default concept graph config
    cfg_cg = OmegaConf.load(cfg.concept_graph_config_path)
    OmegaConf.resolve(cfg_cg)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
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

    # load detection and segmentation models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM(cfg.sam_model_name)  # UltraLytics SAM
    logging.info(f"Load SAM model {cfg.sam_model_name} successful!")

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="/anvme/workspace/v100dd12-3dmem/model/CLIP-ViT-H-14-laion2B-s32B-b79K/open_clip_pytorch_model.bin"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    logging.info(f"Load CLIP model successful!")

    # Initialize the logger
    logger = Logger(
        cfg.output_dir,
        start_ratio,
        end_ratio,
        len(questions_list),
        voxel_size=cfg.tsdf_grid_size,
    )

    # Run all questions
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
        lifelong_context = check_lifelong_memory(lifelong_json_path, lifelong_memory, cfg, question)



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
                # query the VLM for the next navigation point, and the reason for the choice
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
                    lifelong_context=lifelong_context,
                )
                if vlm_response is None:
                    logging.info(
                        f"Question id {question_id} invalid: query_vlm_for_response failed!"
                    )
                    break

                max_point_choice, gpt_answer, n_filtered_snapshots = vlm_response

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
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

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

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, args.start_ratio, args.end_ratio)
