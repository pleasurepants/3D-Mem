import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from src.const import *
import re
import random


client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


def format_content(contents):
    formated_content = []
    image_count = 0
    for c in contents:
        formated_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            image_count += 1
            formated_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    
    # 记录图片数量
    logging.info(f"输入模型的图片总数: {image_count}")
    
    # 检查是否超过限制
    if image_count > 30:
        logging.warning(f"警告：图片数量 ({image_count}) 超过了30张的限制！")
        logging.warning("建议措施：1. 增加预过滤（prefiltering）以减少快照数量，2. 减少 top_k_categories 参数， 3. 调整 vLLM 服务器的 --limit-mm-per-prompt 参数")
        # raise ValueError(f"图片数量 ({image_count}) 超过了vLLM设置的30张限制")
    
    return formated_content


# send information to openai
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
                # model="gpt-4o",  # model = "deployment_name"
                model="qwen",
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
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


# encode tensor images to base64 format
def encode_tensor2base64(img):
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    return img_base64


def format_question(step):
    question = step["question"]
    image_goal = None
    if "task_type" in step and step["task_type"] == "image":
        with open(step["image"], "rb") as image_file:
            image_goal = base64.b64encode(image_file.read()).decode("utf-8")

    return question, image_goal


def get_step_info(step, verbose=False):
    # 1 get question data
    question, image_goal = format_question(step)

    # 2 get step information(egocentric, frontier, snapshot)
    # 2.1 get egocentric views
    egocentric_imgs = []
    if step.get("use_egocentric_views", False):
        for egocentric_view in step["egocentric_views"]:
            egocentric_imgs.append(encode_tensor2base64(egocentric_view))

    # 2.2 get frontiers
    frontier_imgs = []
    for frontier in step["frontier_imgs"]:
        frontier_imgs.append(encode_tensor2base64(frontier))
    frontier_imgs_0 = []
    for frontier in step["frontier_imgs_0"]:
        frontier_imgs_0.append(encode_tensor2base64(frontier))
    frontier_imgs_1 = []
    for frontier in step["frontier_imgs_1"]:
        frontier_imgs_1.append(encode_tensor2base64(frontier))

    # 2.3 get snapshots
    snapshot_classes = {}  # rgb_id -> list of classes
    snapshot_full_imgs = {}  # rgb_id -> full img
    snapshot_crops = {}  # rgb_id -> list of crops
    snapshot_clusters = {}  # rgb_id -> list of clusters
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]["full_img"]
        snapshot_full_imgs[rgb_id] = encode_tensor2base64(snapshot_img)
        snapshot_crops[rgb_id] = [
            encode_tensor2base64(crop_data["crop"])
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        snapshot_class = [
            crop_data["obj_class"]
            for crop_data in step["snapshot_imgs"][rgb_id]["object_crop"]
        ]
        cluster_class = [
            obj_map[int(obj_id)] for obj_id in step["snapshot_objects"][rgb_id]
        ]
        # remove duplicates
        seen_classes.update(sorted(list(set(snapshot_class))))
        snapshot_classes[rgb_id] = snapshot_class
        snapshot_clusters[rgb_id] = cluster_class

    # 3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_full_imgs)))
    keep_index_snapshot = {
        rgb_id: list(range(len(snapshot_crops[rgb_id]))) for rgb_id in snapshot_crops
    }
    if step.get("use_prefiltering") is True:    # seems not set by default
        use_full_obj_list = step["use_full_obj_list"]
        n_prev_snapshot = len(snapshot_full_imgs)
        snapshot_classes, keep_index, keep_index_snapshot = prefiltering(
            question,
            snapshot_classes,
            snapshot_clusters,
            seen_classes,
            step["top_k_categories"],
            image_goal,
            use_full_obj_list,
            verbose=verbose,
        )
        snapshot_full_imgs = {
            rgb_id: snapshot_full_imgs[rgb_id] for rgb_id in keep_index_snapshot.keys()
        }
        for rgb_id in snapshot_classes.keys():
            snapshot_crops[rgb_id] = [
                snapshot_crops[rgb_id][i] for i in keep_index_snapshot[rgb_id]
            ]
        if verbose:
            logging.info(
                f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_full_imgs)}"
            )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        frontier_imgs_0,
        frontier_imgs_1,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        keep_index,
        keep_index_snapshot,
    )


def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    snapshot_crops,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene that is able to observe the surroundings and explore the environment. You are tasked with indoor navigation, and you are required to choose either a Snapshot or a Frontier image to explore and find the target object required in the question.\n"

    content = []
    # 1 here is some basic info
    text = "Definitions:\n"
    text += (
        "Snapshot: A focused observation of several objects. It contains a full image of the cluster of objects, and separate image crops of each object. "
        + "Choosing a snapshot means that the object asked in the question is within the cluster of objects that the snapshot represents, and you will choose that object as the final answer of the question. "
        + "Therefore, if you choose a snapshot, you should also choose the object in the snapshot that you think is the answer to the question.\n"
    )
    text += "Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"

    # 2 here is the question
    text += f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Frontier/Snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 3 here is the egocentric views
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    # 4 here is the snapshot images
    text = "The followings are all the snapshots that you can choose. Following each snapshot image are the class name and image crop of each object contained in the snapshot.\n"
    text += "Please note that the class name may not be accurate due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Snapshot {i} ", snapshot_imgs[rgb_id]))
            for j in range(len(snapshot_crops[rgb_id])):
                content.append(
                    (
                        f"Object {j}: {snapshot_classes[rgb_id][j]}",
                        snapshot_crops[rgb_id][j],
                    )
                )
            content.append(("\n",))

    # 5 here is the frontier images
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))

    # 6 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i, Object j' or 'Frontier i', where i, j are the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the fridge in the first snapshot, please return 'Snapshot 0, Object 2', where 2 is the index of the fridge in that snapshot.\n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    content.append((text,))

    return sys_prompt, content


def format_explore_prompt_snapshot(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    snapshot_crops,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene that is able to observe the surroundings and explore the environment. "
    sys_prompt += "You are tasked with indoor navigation, and you are required to choose a Snapshot to find the target object required in the question.\n"

    content = []
    # 1 here is some basic info
    text = "Definitions:\n"
    text += (
        "Snapshot: A focused observation of several objects. It contains a full image of the cluster of objects, and separate image crops of each object. "
        + "Choosing a snapshot means that the object asked in the question is within the cluster of objects that the snapshot represents, and you will choose that object as the final answer of the question. "
        # + "Therefore, if you choose a snapshot, you should also choose the object in the snapshot that you think is the answer to the question.\n"
        + "You should always try to select a Snapshot and also choose the object in the snapshot that you think is the answer to the question. "
        + "Only if you are absolutely sure that none of the Snapshots contain enough information should you reply with 'No Snapshot is available'.\n"
    )
    # text += "Frontier: An unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction.\n"

    # 2 here is the question
    text += f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 3 here is the egocentric views
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    # 4 here is the snapshot images
    text = "The followings are all the snapshots that you can choose. Following each snapshot image are the class name and image crop of each object contained in the snapshot.\n"
    text += "Please note that the class name may not be accurate due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make the decision.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available\n",))
    else:
        for i, rgb_id in enumerate(snapshot_imgs.keys()):
            content.append((f"Snapshot {i} ", snapshot_imgs[rgb_id]))
            for j in range(len(snapshot_crops[rgb_id])):
                content.append(
                    (
                        f"Object {j}: {snapshot_classes[rgb_id][j]}",
                        snapshot_crops[rgb_id][j],
                    )
                )
            content.append(("\n",))

    # # 5 here is the frontier images
    # text = "The followings are all the Frontiers that you can explore: \n"
    # content.append((text,))
    # if len(frontier_imgs) == 0:
    #     content.append(("No Frontier is available\n",))
    # else:
    #     for i in range(len(frontier_imgs)):
    #         content.append((f"Frontier {i} ", frontier_imgs[i]))
    #         content.append(("\n",))
    
    ## ---- 枚举所有可用index
    # if len(snapshot_imgs) > 0:
    #     indices_list = ", ".join([str(i) for i in range(len(snapshot_imgs))])
    #     # 组合所有可选格式
    #     example_str = "', '".join([f"Snapshot {i}" for i in range(len(snapshot_imgs))])
    #     indices_hint = f"The only available Snapshot indices are: {indices_list}.\n"
    #     indices_example = f"You can answer using only '{example_str}', but never use an index not in this list.\n"
    # else:
    #     indices_hint = ""
    #     indices_example = ""

    # 6 here is the format of the answer
    text = "Please answer in exactly one of the following two formats:\n"
    text += "1. Snapshot i, Object j\n[Your reason for your choice.]\n"
    text += "2. No Snapshot is available.\n"
    text += "The two formats are mutually exclusive. Never combine 'No Snapshot is available' with any Snapshot index.\n"
    text += "If you select a Snapshot, please provide your answer in the following format: 'Snapshot i, Object j', where i, j are the index of the snapshot and the object you choose. "
    text += "For example, if you choose the fridge in the first snapshot, please return 'Snapshot 0, Object 2', where 2 is the index of the fridge in that snapshot.\n"
    text += "You can explain the reason for your choice, but put it in a new line after the choice.\n"
    text += "If you are absolutely sure that none of the Snapshots contain enough information, please return 'No Snapshot is available'.\n"
    text += "You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot.\n"
    text += "Only use the provided Snapshot indices, and DO NOT make up any index that is not listed above."
    content.append((text,))

    return sys_prompt, content


def format_explore_prompt_frontier(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    # snapshot_crops,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    context=None,
    episodic_con=None,
    frontier_type="BVF",
):
    """
    Frontier-selection prompt with explicit Step 0/1/2/3 and FINAL:
      - Clear semantics (Frontier / Episodic / Experience).
      - Optional blocks via has_episodic / has_experience.
      - Strict index-only discipline.
      - Reason first, answer last; FINAL line prints only: 'frontier i'.
    """
    
    # --------- Presence flags ----------
    has_episodic = bool(episodic_con and isinstance(episodic_con, str) and episodic_con.strip())
    has_experience = bool(context and isinstance(context, str) and context.strip())
    has_ego = bool(egocentric_view and egocentric_imgs and len(egocentric_imgs) > 0)
    
    # =========================
    # System role & definitions (based on user's template)
    # =========================
    label_word = "BVF" if str(frontier_type).upper() == "BVF" else "CVF"
    sys_prompt = ""
    sys_prompt += (
        "You are an embodied agent for exploration in an indoor environment to find the target object required in the question. "
        + "At each step of exploration, you will be given frontier snapshots of your surrounding environment; your task is to pick EXACTLY ONE frontier to move to for further exploration or solving the question.\n\n"
        + "FRONTIERs are candidate entry points toward yet-unseen or information-rich regions—typical visual patterns include doorways/thresholds, corridors/intersections, stairs, corners/turns, or vantage points that likely open new coverage.\n\n"
        + "You will be given 2 types of frontiers: Broad-View Frontier (BVF) segments your 360° surrounding environment so that you can have an overview. "
        + "You SHALL pick EXACTLY ONE BVF to look closer. With the selected BVF, you DO NOT move; you further break down that direction into Closer-View Frontiers (CVF), which give narrowed perspectives. "
        + "You SHALL pick EXACTLY ONE CVF to move to in the next step.\n\n"
        + "You will be given the following information as contexts:\n"
        + "EGOCENTRIC VIEW (if shown): The agent’s immediate forward-looking camera view; use it as local context only.\n"
        + "EPISODIC CONTEXT (if present): A factual textual summary of the previous steps within THIS episode (visited path, observations, likely-unseen areas). "
        + "Use this to avoid redundancy and prefer novel, informative directions. It is evidence, not a command.\n"
        + "EXPERIENCE REPLAY (if present): A textual experience of frontier selection to solve a similar question in a similar environment—how the decision was made, which frontier was chosen, what actions followed, the outcome/reward, a brief critique, and an abstraction to reflect on.\n\n"
        + "RULES:\n"
        + "- You will only be given either BVFs or CVFs at a time (BVF for looking closer; CVF for moving next).\n"
        + "- Your reasoning must be concrete and visual. Name specific objects, layouts, textures, lighting, text-bearing surfaces/symbols, and any cues directly relevant to the question.\n"
        + "- You must select one of the provided candidates; do NOT output that none is suitable.\n"
        + f"- Output the rationale first and the answer last. On the final line, print ONLY '{label_word} i' (the chosen index).\n"
    )
    
    content = []
    # =========================
    # Frontier candidates
    # =========================
    content.append((f"You are given the following frontiers ({frontier_type} only at this step):",))
    if len(frontier_imgs) == 0:
        content.append(("No frontier is available.",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"{label_word} {i}", frontier_imgs[i]))

    # =========================
    # Egocentric (optional)
    # =========================
    if has_ego:
        content.append(("Egocentric forward view (immediate local context):", egocentric_imgs[-1]))

    # =========================
    # Episodic context (optional)
    # =========================
    if has_episodic:
        content.append((
            "Episodic context — summary of the recent steps within THIS episode (path you followed, what you observed, "
            "what seems already covered vs. still unexplored). Use this to avoid redundancy and to prefer novel, decision-relevant directions:\n"
            + episodic_con.strip(),
        ))
    
    # =========================
    # Experience replay (optional)
    # =========================
    if has_experience:
        content.append((
            "Experience replay — knowledge from OTHER episodes in similar scenes. It may include 'Critique:' (what happened) and 'Abstraction:' (a simple rule). "
            "Use the Abstraction as a transferable hint for this scene. Extract only transferable visual patterns/strategies and prefer the current visible evidence when conflicts arise:\n"
            + context.strip(),
        ))
    
    # =========================
    # Question
    # =========================
    q_text = f"Now you need to answer the question: {question}"
    if image_goal is not None:
        content.append((q_text, image_goal))
    else:
        content.append((q_text + " ",))
        
    # =========================
    # Minimal reasoning scaffold consistent with user's instruction
    # =========================
    guidance = (
        "IMPORTANT: You MUST reason step by step using ONLY the provided frontiers (Step 1, Step 2, ...), and do NOT skip steps. "
        "Use the contexts (EPISODIC/EXPERIENCE) if present to avoid redundancy and transfer useful cues. "
        f"Output the rationale first and the answer last. On the final line, print ONLY '{label_word} i'."
    )
    content.append((guidance,))

    return sys_prompt, content


def format_prefiltering_prompt(question, class_list, top_k=10, image_goal=None):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene.\n"
    prompt = "Your goal is to answer questions about the scene through exploration.\n"
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance.\n"
    prompt += "These are the rules for the task.\n"
    prompt += "1. Read through the whole object list.\n"
    prompt += "2. Rank objects in the list based on how well they can help your exploration given the question.\n"
    prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
    prompt += "4. Do not print any object not included in the list or include any additional information in your response.\n"
    content.append((prompt,))
    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects:\n"
    prompt += "Question: What can I use to watch my favorite shows and movies?\n"
    prompt += (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    prompt += "painting\nspeaker\nbox\ncabinet\nlamp\ntv\nbook rack\nsofa\noven\nbed\ncurtain\n"
    prompt += "Answer: tv\nspeaker\nsofa\nbed\n"
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order:\n"
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append(("\n",))
    else:
        content.append((prompt + "\n",))
    prompt = (
        "Following is a list of objects that you can choose, each object one line\n"
    )
    for i, cls in enumerate(class_list):
        prompt += f"{cls}\n"
    prompt += "Answer: "
    content.append((prompt,))
    return sys_prompt, content


def get_prefiltering_classes(question, seen_classes, top_k=10, image_goal=None):
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    response = call_openai_api(prefiltering_sys, prefiltering_content)
    if response is None:
        return []

    # parse the response and return the top_k objects
    selected_classes = response.strip().split("\n")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes

## in aeqa, change '\n' to ' '
def prefiltering(
    question,
    snapshot_classes,
    snapshot_clusters,
    seen_classes,
    top_k=10,
    image_goal=None,
    use_full_obj_list=False,
    verbose=False,
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")

    keep_index = [
        i
        for i, k in enumerate(snapshot_clusters.keys())
        if len(set(snapshot_clusters[k]) & set(selected_classes)) > 0
    ]
    keep_snapshot_id = [list(snapshot_classes.keys())[i] for i in keep_index]
    snapshot_classes = {rgb_id: snapshot_classes[rgb_id] for rgb_id in keep_snapshot_id}

    keep_index_snapshot = {}
    for rgb_id in keep_snapshot_id:
        keep_index_snapshot[rgb_id] = [
            i
            for i in range(len(snapshot_classes[rgb_id]))
            if snapshot_classes[rgb_id][i] in selected_classes
        ]
        snapshot_classes[rgb_id] = [
            snapshot_classes[rgb_id][i] for i in keep_index_snapshot[rgb_id]
        ]

    return snapshot_classes, keep_index, keep_index_snapshot


def simple_recall_and_aggregate(frontier_imgs_b64, cfg, exclude_question_id=None, top_k=1, strategy: str = 'sim', current_question: str = None, rrf_k: int = 60):
    pass


def clean_reason(reason):
    """
    更鲁棒地去除reason/answer中带有 [answer: xxx] 或 [reason: xxx] 及所有[]，只保留核心文本
    """
    # 去掉开头类似于 [answer: xxxx] 或 [reason: xxx] 的内容（忽略大小写）
    reason = re.sub(r'^\s*\[\s*(answer|reason)\s*:\s*([^\]]+)\]\s*', r'\2', reason, flags=re.IGNORECASE)
    # 再去掉所有剩余的 []
    reason = reason.replace('[', '').replace(']', '')
    # 去除首尾引号和空格
    reason = reason.strip().strip("\"'")
    return reason


def save_base64_to_png(b64_str, save_dir, step_idx, idx):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{step_idx}-frontier{idx}.png")
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes))
    img.save(save_path)
    return save_path


def save_base64_to_png_layer1(b64_str, save_dir, step_idx, idx, idx0):
    os.makedirs(save_dir, exist_ok=True)
    idx = idx%3 # less than 3 clusters in one layer1
    save_path = os.path.join(save_dir, f"{step_idx}-frontier{idx0}_{idx}.png")
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes))
    img.save(save_path)
    return save_path


def frontier_context(
    folder,
    question="Please summarize the agent's exploration so far.",
    max_num=5
):
    files = [f for f in os.listdir(folder) if f.endswith('.png')]
    if not files:
        recent_imgs = []
    else:
        def extract_step_idx(f):
            name = os.path.splitext(f)[0]
            # parts = name.split('-')
            # step = int(parts[0])
            # fidx = '-'.join(parts[1:])  # 保留为字符串
            # return (step, fidx)
            pattern = r'task-(\d+)_step-(\d+)_?(.*)'
            match = re.match(pattern, name)
            if match:
                subtask_id = int(match.group(1))
                step = int(match.group(2))
                fidx = match.group(3)
                return (subtask_id, step, fidx)
        files = sorted(files, key=extract_step_idx, reverse=True)[:max_num]
        files = sorted(files, key=extract_step_idx)
        recent_imgs = []
        for f in files:
            with open(os.path.join(folder, f), 'rb') as imgf:
                img_b64 = base64.b64encode(imgf.read()).decode('utf-8')
            recent_imgs.append( (f, img_b64) )
    
    # ===== sys_prompt 一行一行拼接 =====
    sys_prompt = ""
    sys_prompt += "You are an agent navigating an indoor environment. "
    sys_prompt += "The following images represent the sequence of directions or regions the agent has chosen to explore, in order. "
    sys_prompt += "Your job is to write a concise context summary that describes: "
    sys_prompt += "(1) Which areas or room types the agent has already explored (based on the sequence); "
    sys_prompt += "(2) Which areas or directions may remain unexplored or uncertain; "
    sys_prompt += "(3) Any useful patterns or observations about the current state. "
    sys_prompt += "Do NOT make a decision for the next move. Do NOT output action suggestions. "
    sys_prompt += "The output should be a short, objective summary paragraph for use as context in later decision-making. "
    sys_prompt += "Please pay attention to the order of the images, as they represent the exploration path."
    
    content = []
    
    # 1. 问题描述
    text = ""
    text += "Exploration summary request: "
    text += question
    content.append((text,))
    
    # 2. Example/example output
    text = ""
    text += "Example: "
    text += "The agent has explored a kitchen area and a hallway leading to a living room. "
    text += "The bathroom and a side room to the right have not been explored yet. "
    text += "Most of the agent's trajectory has covered open spaces, with some doors and closed areas remaining unexplored."
    content.append((text,))

    # 3. 图片有序拼接
    text = ""
    text += "Below are the most recent selected exploration directions, in order (earliest to latest): "
    content.append((text,))

    for i, (fname, img_b64) in enumerate(recent_imgs):
        text = ""
        text += f"Step {i+1}: chosen direction ({fname}). "
        content.append((text, img_b64))

    # 4. 明确只输出context summary，不要建议
    text = ""
    text += "Please output ONLY a single paragraph context summary, similar to the example above. "
    text += "Do NOT make suggestions or give next-step decisions."
    content.append((text,))

    return sys_prompt, content


def parse_frontier_index(output: str):
    """
    解析输出文本，返回(reason, index)
    支持全文任意位置的frontier index格式
    """
    # 支持 'frontier i'、'bvf i'、'cvf i' 三种格式（取最后一个命中）
    matches = list(re.finditer(r'(?:frontier|bvf|cvf)\s*(\d+)', output, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        index = int(last_match.group(1))
        reason = output[:last_match.start()].strip()
        return reason, index
    else:
        raise ValueError(f"Could not parse frontier index")


def _shorten(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    t = text.strip()
    return (t[:max_len] + " ...") if len(t) > max_len else t

def aggregate_recall_contexts_for_layer(
    layer_alias: str,        # 用自然词，比如 "initial directions" / "closer looks"
    contexts: list,          # List[Optional[str]]，与候选对齐
    indices: Optional[list] = None,  # 若只汇总子集（例如某个方向下的 closer looks），则传对应的全局索引
) -> Optional[str]:
    """
    生成一段自然语言总述：
    - 先是层级总述（不用出现 'layer' 字样）
    - 后面按候选编号列出每个候选的一句话摘要（截断）
    """
    items = []
    if indices is None:
        pairs = list(enumerate(contexts))
    else:
        pairs = [(i, contexts[i]) for i in indices]

    for i, ctx in pairs:
        if ctx:
            items.append(f"(candidate #{i}) { _shorten(ctx, 300) }")

    if not items:
        return None

    header = (
        f"Recalled summary for {layer_alias}: "
        f"the following candidates have useful past hints that may guide the decision."
    )
    return header + "\n" + "\n".join(items)
    

def explore_step(step, cfg, verbose=False, chosen_frontier_path=None, step_idx=None):
    step["use_prefiltering"] = cfg.prefiltering # true
    step["top_k_categories"] = cfg.top_k_categories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        frontier_imgs_0,
        frontier_imgs_1,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        snapshot_id_mapping,
        snapshot_crop_mapping,
    ) = get_step_info(step, verbose)
    
    ## ==== Step 1: snapshot prompt ====
    # sys_prompt, content = format_explore_prompt(
    sys_prompt, content = format_explore_prompt_snapshot(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_full_imgs,
        snapshot_classes,
        snapshot_crops,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
    )

    if verbose:
        # logging.info(f"Input prompt:")
        logging.info(f"Input prompt (snapshot):")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    if len(snapshot_full_imgs) == 0:
        print("No snapshot images available, directly entering frontier exploration.")
    else:
        print(f"Snapshot images available: {len(snapshot_full_imgs)}")
        retry_bound = 3
        final_response = None
        final_reason = None
        for _ in range(retry_bound):
            full_response = call_openai_api(sys_prompt, content)
            if full_response is None:
                print("call_openai_api (snapshot) returns None, retrying")
                continue
            if isinstance(full_response, list):
                full_response = " ".join(full_response)
            full_response = full_response.strip().lower()
            if full_response.startswith("snapshot"):
                # tokens = full_response.split()
                # if len(tokens) >= 2 and tokens[1].isdigit():
                #     idx = int(tokens[1])
                #     reason = clean_reason(" ".join(tokens[2:]).strip()) 
                #     if 0 <= idx < len(snapshot_full_imgs) and reason != "":
                #         response = f"{tokens[0]} {tokens[1]}"
                #         # reason = " ".join(tokens[2:]).strip()
                #         # reason = clean_reason(reason)  
                #         return response, snapshot_id_mapping, reason, len(snapshot_full_imgs)
                #     elif 0 <= idx < len(snapshot_full_imgs) and reason == "":
                #         print(f"Snapshot index {tokens[1]} has no reason.")
                #         continue
                #     else:
                #         print(f"Snapshot index out of range: {tokens[1]}")
                #         continue
                # 解析 full_response，提取 "snapshot i, object j" 以及后续解释
                match = re.match(r"snapshot\s*(\d+)\s*,\s*object\s*(\d+)[,.\n]?(.*)", full_response, re.IGNORECASE | re.DOTALL)
                if match:
                    snapshot_idx = int(match.group(1))
                    object_idx = int(match.group(2))
                    reason = match.group(3).strip()
                    # 检查索引是否有效
                    if 0 <= snapshot_idx < len(snapshot_full_imgs):
                        # snapshot_id_mapping 需要在外部定义，这里假设已存在
                        response = f"snapshot {snapshot_idx}, object {object_idx}"
                        final_response = response
                        final_reason = reason
                        # 返回或保存结果
                        return final_response, snapshot_id_mapping, snapshot_crop_mapping, final_reason, len(snapshot_full_imgs)
                    else:
                        print(f"Snapshot index out of range: {snapshot_idx} (length: {len(snapshot_full_imgs)})")
                        continue
                else:
                    print(f"Snapshot response format error: {full_response}")
                    continue
            elif "no snapshot is available" in full_response:
                # 明确拒绝，直接进入frontier
                break
            else:
                print(f"Unrecognized snapshot response: {full_response}")
                continue
        
    # retry_bound = 3
    # final_response = None
    # final_reason = None
    # for _ in range(retry_bound):
    #     response = call_openai_api(sys_prompt, content)

    #     if response is None:
    #         print("call_openai_api returns None, retrying")
    #         continue

    #     response = response.strip()
    #     if "\n" in response:
    #         response = response.split("\n") # "Snapshot i, Object j\nreason" or "Frontier i\nreason"
    #         response, reason = response[0], response[-1]
    #     else:
    #         reason = ""
    #     response = response.lower()
    #     try:
    #         choice_type, choice_id = response.split(",")[0].strip().split(" ")
    #     except Exception as e:
    #         print(f"Error in splitting response: {response}")
    #         print(e)
    #         continue

    #     response_valid = False
    #     if (
    #         choice_type == "snapshot"
    #         and choice_id.isdigit()
    #         and 0 <= int(choice_id) < len(snapshot_full_imgs)
    #     ):
    #         try:
    #             object_choice_type, object_choice_id = (
    #                 response.split(",")[1].strip().split(" ")
    #             )
    #         except Exception as e:
    #             print(f"Error in splitting response: {response}")
    #             print(e)
    #             continue
    #         if (
    #             object_choice_type == "object"
    #             and object_choice_id.isdigit()
    #             and 0
    #             <= int(object_choice_id)
    #             < len(list(snapshot_crop_mapping.values())[int(choice_id)])
    #         ):
    #             response_valid = True
    #     elif (
    #         choice_type == "frontier"
    #         and choice_id.isdigit()
    #         and 0 <= int(choice_id) < len(frontier_imgs)
    #     ):
    #         response_valid = True

    #     if response_valid:
    #         final_response = response
    #         final_reason = reason
    #         break

    # return (
    #     final_response,
    #     snapshot_id_mapping,
    #     snapshot_crop_mapping,
    #     final_reason,
    #     len(snapshot_full_imgs),
    # )
    
    ## ==== Step 2: two-stage frontier prompt ====
    retry_bound = 3

    # ==== (NEW) Layer-0 回忆与聚合（只对初始方向层做） ====
    _replay_top = int(getattr(cfg, "replay_top", 1))
    if _replay_top > 0:
        # 使用本文件中的最简实现（不依赖 context_generator）
        step["replay_layer0_aggregated_context"] = simple_recall_and_aggregate(
            frontier_imgs_b64=frontier_imgs_0,
            cfg=cfg,
            exclude_question_id=step.get("question_id"),
            top_k=_replay_top,
            strategy=(
                "random" if getattr(cfg, "replay_mode", "sim") == "random" else "sim"
            ),
            current_question=question,
        )
    else:
        step["replay_layer0_aggregated_context"] = None
        logging.info("[ReplayCtx] replay_top=0; skip layer0 recall and env context injection.")
        
    episodic_con = None
    if not os.path.exists(chosen_frontier_path):
        os.makedirs(chosen_frontier_path, exist_ok=True)

    png_files = [f for f in os.listdir(chosen_frontier_path) if f.endswith('.png')]
    if len(png_files) > 0:
        sys_prompt, content = frontier_context(chosen_frontier_path)
        episodic_con = call_openai_api(sys_prompt, content)
        logging.info(f"Froncon label: {episodic_con}")
    else:
        pass
    
    layer0_con = step.get("replay_layer0_aggregated_context") if _replay_top > 0 else None
    try:
        logging.info(f"[ReplayCtx] layer0 aggregated context len: {len(layer0_con) if isinstance(layer0_con, str) else 'None'}")
    except Exception:
        pass
    
    ## ==== Step 2.1: 先让VLM在layer0大簇里选 ====
    sys_prompt, content = format_explore_prompt_frontier(
        question,
        egocentric_imgs,
        frontier_imgs_0,
        snapshot_full_imgs,
        snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        context=layer0_con, # TODO: !!!
        episodic_con=episodic_con,  # TODO: !!!
        frontier_type="BVF",
    )
    if verbose:
        try:
            has_context = bool(layer0_con and isinstance(layer0_con, str) and layer0_con.strip())
            logging.info(f"[PromptDebug] frontier layer0 has_context={has_context}")
        except Exception:
            pass
    if verbose:
        logging.info(f"Input prompt (frontier layer0):")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    idx0 = None
    for _ in range(retry_bound):
        full_response = call_openai_api(sys_prompt, content)
        if full_response is None:
            print("call_openai_api (frontier layer0) returns None, retrying")
            continue
        if isinstance(full_response, list):
            full_response = " ".join(full_response)
        full_response = full_response.strip().lower()
        try:
            reason, idx0 = parse_frontier_index(full_response)
            if 0 <= idx0 < len(frontier_imgs_0):
                break
            else:
                print(f"Layer0 index out of range: {idx0} (length: {len(frontier_imgs_0)})")
        except Exception as e:
            print(f"Layer0 format error: {full_response} | {e}")
    
    if idx0 is None:
        idx_random = random.randrange(0, max(1, len(frontier_imgs_0)))
        response = f'frontier {idx_random}'
        reason = "no valid index found, randomly selected one."
        return response, snapshot_id_mapping, snapshot_crop_mapping, reason, len(snapshot_full_imgs)
    logging.info(f"[Layer0] VLM selected index: {idx0}")
    logging.info(f"reason for layer0 selection: {reason}")
    for k, v in step['layer0_to_layer1'].items():
        logging.info(f"  Layer0 {k}: {v}")
        
    ## ==== Step 2.2: 在选中的layer0大簇下所有layer1细簇中选 ====
    full_response_layer0 = full_response.strip().lower()
    if idx0 not in step['layer0_to_layer1']:
        response = f"frontier {idx0}"
        final_reason = full_response.lower()
        logging.info(f"[Layer0] Layer0 index {idx0} has no corresponding layer1 subclusters. Directly returning layer0 as the frontier (global index: {idx0})")
        return response, snapshot_id_mapping, snapshot_crop_mapping, final_reason, len(snapshot_full_imgs)
    else:
        layer1_indices = step['layer0_to_layer1'][idx0]   # 例如 [1, 2], can be like [5,6,7], but in prompt changed to [0,1,2]
        frontier_imgs_subgroup = [frontier_imgs_1[i] for i in layer1_indices]
        if len(layer1_indices) == 1:
            final_layer1_idx = layer1_indices[0]
            global_frontier_idx = len(step["frontier_imgs_0"]) + final_layer1_idx
            response = f"frontier {global_frontier_idx}"
            final_reason = "Only one candidate in this subcluster, selected by default."
            logging.info(f"[Layer1] Only one candidate ({global_frontier_idx}), selected by default.")
            # save_base64_to_png(frontier_imgs_1[int(final_layer1_idx)], chosen_frontier_path, step_idx, final_layer1_idx)
            save_base64_to_png_layer1(frontier_imgs_1[int(final_layer1_idx)], chosen_frontier_path, step_idx, final_layer1_idx, idx0)
            return response, snapshot_id_mapping, snapshot_crop_mapping, final_reason, len(snapshot_full_imgs)
        
        ## ==== 对该方向下的“更近处视角”子集做聚合（indices 为全局 layer1 索引） ====
        layer1_texts_all = step.get("replay_context_text_per_frontier", {}).get("layer1", [])   # TODO: no such key
        layer1_context_text = None
        if layer1_texts_all and isinstance(layer1_indices, list) and len(layer1_indices) > 0:
            layer1_context_text = aggregate_recall_contexts_for_layer(
                layer_alias="closer looks",
                contexts=layer1_texts_all,
                indices=layer1_indices
            )
        
        ## ==== (NEW) 对该方向的更近处子集做回忆与聚合 ====
        if _replay_top > 0:
            # 根据选择的大簇子集做同样的最简实现
            subgroup_b64 = [frontier_imgs_1[i] for i in layer1_indices]
            layer1_context_text = simple_recall_and_aggregate(
                frontier_imgs_b64=subgroup_b64,
                cfg=cfg,
                exclude_question_id=step.get("question_id"),
                top_k=_replay_top,
                strategy=(
                    "random" if getattr(cfg, "replay_mode", "sim") == "random" else "sim"
                ),
                current_question=question,
            )
        else:
            layer1_context_text = None
            logging.info("[ReplayCtx] replay_top=0; skip layer1 subgroup recall and env context injection.")
        
        sys_prompt, content = format_explore_prompt_frontier(
            question,
            egocentric_imgs,
            frontier_imgs_subgroup,
            snapshot_full_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            context=(layer1_context_text if _replay_top > 0 else None),
            episodic_con=episodic_con,
            frontier_type="CVF",
        )
        if verbose:
            try:
                has_context_l1 = bool(layer1_context_text and isinstance(layer1_context_text, str) and layer1_context_text.strip())
                logging.info(f"[PromptDebug] frontier layer1 has_context={has_context_l1}")
            except Exception:
                pass
        if verbose:
            logging.info(f"Input prompt (frontier layer1):")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        idx1_in_subgroup = None
        final_reason = ""
        
        for _ in range(retry_bound):
            full_response = call_openai_api(sys_prompt, content)
            if full_response is None:
                print("call_openai_api (frontier layer1) returns None, retrying")
                continue
            if isinstance(full_response, list):
                full_response = " ".join(full_response)
            full_response = full_response.strip().lower()
            try:
                reason, idx1_in_subgroup = parse_frontier_index(full_response)
                if 0 <= idx1_in_subgroup < len(frontier_imgs_subgroup):
                    # 可以顺便保留推理部分（比如取出最后一行前的内容，作为reason）
                    # 这里你原来是用 group(2) 取 reason，可以保留
                    lines = [line.strip() for line in full_response.strip().split('\n') if line.strip()]
                    if len(lines) > 1:
                        final_reason = "\n".join(lines[:-1])
                    else:
                        final_reason = ""
                    break
                else:
                    print(f"Layer1 index out of range: {idx1_in_subgroup} (length: {len(frontier_imgs_subgroup)})")
            except Exception as e:
                print(f"Layer1 format error: {full_response} | {e}")
    
    if idx1_in_subgroup is None:
        idx_random = random.randrange(0, max(1, len(frontier_imgs_subgroup)))
        # 映射回全局 layer1 索引
        final_layer1_idx = layer1_indices[idx_random]
        global_frontier_idx = len(step["frontier_imgs_0"]) + final_layer1_idx
        response = f'frontier {global_frontier_idx}'
        reason = f"Randomly selected index {global_frontier_idx} due to parsing failure."
        return response, snapshot_id_mapping, snapshot_crop_mapping, reason, len(snapshot_full_imgs)
        
    elif idx1_in_subgroup >= len(layer1_indices):
        logging.warning(f"[Fallback] Invalid or missing Layer1 index ({idx1_in_subgroup}), fallback to Layer0 index {idx0}")
        response = f"frontier {idx0}"
        final_reason = full_response_layer0
        return response, snapshot_id_mapping, snapshot_crop_mapping, full_response_layer0, len(snapshot_full_imgs)
        
    final_layer1_idx = layer1_indices[idx1_in_subgroup]
    # frontier index = len(self.frontiers_layer0) + final_layer1_idx
    global_frontier_idx = len(step["frontier_imgs_0"]) + final_layer1_idx
    response = f"frontier {global_frontier_idx}"
    logging.info(f"[Layer1] VLM selected group index: {idx1_in_subgroup}")
    logging.info(f"[Layer1] This corresponds to global layer1 index: {final_layer1_idx} (global index: {global_frontier_idx})")

    save_base64_to_png_layer1(frontier_imgs_1[int(final_layer1_idx)], chosen_frontier_path, step_idx, final_layer1_idx, idx0)

    return response, snapshot_id_mapping, snapshot_crop_mapping, reason, len(snapshot_full_imgs)