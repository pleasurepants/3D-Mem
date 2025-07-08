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
import base64
from PIL import Image
from io import BytesIO


client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


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
                model="minicpml",  # gpt-4o
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

    # 2.3 get snapshots
    snapshot_imgs, snapshot_classes = [], []
    obj_map = step["obj_map"]
    seen_classes = set()
    for i, rgb_id in enumerate(step["snapshot_imgs"].keys()):
        snapshot_img = step["snapshot_imgs"][rgb_id]
        snapshot_imgs.append(encode_tensor2base64(snapshot_img))
        snapshot_class = [obj_map[int(sid)] for sid in step["snapshot_objects"][rgb_id]]
        # remove duplicates
        snapshot_class = sorted(list(set(snapshot_class)))
        seen_classes.update(snapshot_class)
        snapshot_classes.append(snapshot_class)

    # 3 prefiltering, note that we need the obj_id_mapping
    keep_index = list(range(len(snapshot_imgs)))
    if step.get("use_prefiltering") is True:
        n_prev_snapshot = len(snapshot_imgs)
        snapshot_classes, keep_index = prefiltering(
            question,
            snapshot_classes,
            seen_classes,
            step["top_k_categories"],
            image_goal,
            verbose,
        )
        snapshot_imgs = [snapshot_imgs[i] for i in keep_index]
        if verbose:
            logging.info(
                f"Prefiltering snapshot: {n_prev_snapshot} -> {len(snapshot_imgs)}"
            )

    return (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        keep_index,
    )


def format_explore_prompt(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a Snapshot as the answer or a Frontier to further explore. "
    sys_prompt += "Definitions: "
    sys_prompt += "Snapshot: A focused observation of several objects. Choosing a Snapshot means that this snapshot image contains enough information for you to answer the question. "
    sys_prompt += "If you choose a Snapshot, you need to directly give an answer to the question. If you don't have enough information to give an answer, then don't choose a Snapshot. "
    sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction. "
    sys_prompt += "If you choose a Frontier, you need to explain why you would like to choose that direction to explore. "

    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append((" ",))
    else:
        content.append((text + " ",))

    text = "Select the Frontier/Snapshot that would help find the answer of the question. "
    content.append((text,))

    # 2 add egocentric view
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append((" ",))

    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can choose (followed with contained object classes) "
    text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make decisions. "
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available ",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"Snapshot {i} ", snapshot_imgs[i]))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append((" ",))

    # 4 here is the frontier images
    text = "The followings are all the Frontiers that you can explore:  "
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available ",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append((" ",))

    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i [Answer]' or 'Frontier i [Reason]', where i is the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the first snapshot, you can return 'Snapshot 0 The fruit bowl is on the kitchen counter.'. "
    text += "If you choose the second frontier, you can return 'Frontier 1 I see a door that may lead to the living room.'. "
    text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; You must only choose from the provided Snapshot or Frontier indices. Do not make up an index that is not listed above."
    text += "(2) you can also utilize other snapshots, frontiers and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question. You must only choose from the provided Snapshot or Frontier indices. Do not make up an index that is not listed above. "
    content.append((text,))

    return sys_prompt, content


def format_explore_prompt_frontier(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    context=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose a Frontier to further explore. "
    sys_prompt += "Definitions: "
    sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction. "
    sys_prompt += "If you choose a Frontier, you need to explain why you would like to choose that direction to explore. "
    if context:
        sys_prompt += "Context: The following summary describes the agent's past exploration and current known status. Use this context to help you make a better choice, but do not treat it as a direct instruction.\n"
        sys_prompt += f"{context}\n"



    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append((" ",))
    else:
        content.append((text + " ",))

    text = "Select the Frontier that would help find the answer of the question. "
    content.append((text,))

    # 2 add egocentric view
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append((" ",))


    # 4 here is the frontier images
    text = "The followings are all the Frontiers that you can explore:  "
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append((" ",))


    text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index you choose. "
    text += "You must select one of the provided Frontier indices. Choose the frontier most likely to lead to the answer, and briefly explain why it is helpful for answering the question. "
    text += "For example: 'Frontier 1 There is a door that may lead to the kitchen, where the answer might be found.' "
    text += "Only use the provided indices. Do not make up new indices."

    
    content.append((text,))


    content.append((text,))

    return sys_prompt, content




def format_explore_prompt_snapshot(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    ):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. "
    sys_prompt += "To answer the question, you are required to choose a Snapshot and provide your answer based on it. "
    sys_prompt += "Definitions: "
    sys_prompt += "Snapshot: A focused observation of several objects. Choosing a Snapshot means that this snapshot image contains enough information for you to answer the question. "
    sys_prompt += "You should always try to select a Snapshot and answer the question directly based on the information it provides. "
    sys_prompt += "Only if you are absolutely sure that none of the Snapshots contain enough information should you reply with 'No Snapshot is available'."
    # sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction. "
    # sys_prompt += "If you choose a Frontier, you need to explain why you would like to choose that direction to explore. "

    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append((" ",))
    else:
        content.append((text + " ",))

    text = "Select the Snapshot that would help find the answer of the question. "
    content.append((text,))

    # 2 add egocentric view
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append((" ",))

    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can choose (followed with contained object classes) "
    text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make decisions. "
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No Snapshot is available",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"Snapshot {i} ", snapshot_imgs[i]))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append((" ",))


    text = "Please provide your answer in the following format: 'Snapshot i [Answer]' or 'No Snapshot is available', where i is the index of the snapshot you choose. "
    text += "You should select one of the provided Snapshots and give a clear and direct answer to the question. Only reply 'No Snapshot is available' if it is truly impossible to answer from any Snapshot. "
    text += "Write your answer as a complete sentence that directly responds to the question, not just a description of the image. Use simple and direct sentences, avoid vague or descriptive language. Do not mention words like 'snapshot', 'on the left of the image', etc. "
    text += "For example, if you choose the first snapshot, you can return 'Snapshot 0 The fruit bowl is on the kitchen counter.'. "
    text += "or if you choose the second snapshot, you can return 'Snapshot 1 Next to the fireplace'. "
    text += "You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot."
    text += "Note: Do not mention words like 'snapshot', 'in the image', or image positions. Only use the provided Snapshot indices, and do not make up any index that is not listed above. Only output the complete answer as a direct response, without any extra words, explanation, or reasoning."


    content.append((text,))

    return sys_prompt, content









def format_prefiltering_prompt(question, class_list, top_k=10, image_goal=None):
    content = []
    sys_prompt = "You are an AI agent in a 3D indoor scene. "
    prompt = "Your goal is to answer questions about the scene through exploration. "
    prompt += "To efficiently solve the problem, you should first rank objects in the scene based on their importance. "
    prompt += "These are the rules for the task. "
    prompt += "1. Read through the whole object list. "
    prompt += "2. Rank objects in the list based on how well they can help your exploration given the question. "
    prompt += f"3. Reprint the name of all objects that may help your exploration given the question. "
    prompt += "4. Do not print any object not included in the list or include any additional information in your response. "
    content.append((prompt,))
    # ------------------format an example-------------------------
    prompt = "Here is an example of selecting helpful objects: "
    prompt += "Question: What can I use to watch my favorite shows and movies? "
    prompt += (
        "Following is a list of objects that you can choose, each object one line "
    )
    prompt += "painting speaker box cabinet lamp tv book rack sofa oven bed curtain "
    prompt += "Answer: tv speaker sofa bed "
    content.append((prompt,))
    # ------------------Task to solve----------------------------
    prompt = f"Following is the concrete content of the task and you should retrieve helpful objects in order: "
    prompt += f"Question: {question}"
    if image_goal is not None:
        content.append((prompt, image_goal))
        content.append((" ",))
    else:
        content.append((prompt + " ",))
    prompt = (
        "Following is a list of objects that you can choose, each object one line "
    )
    for i, cls in enumerate(class_list):
        prompt += f"{cls} "
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
    selected_classes = response.strip().split(" ")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes


def prefiltering(
    question, snapshot_classes, seen_classes, top_k=10, image_goal=None, verbose=False
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal
    )
    if verbose:
        logging.info(f"Prefiltering selected classes: {selected_classes}")

    keep_index = [
        i
        for i in range(len(snapshot_classes))
        if len(set(snapshot_classes[i]) & set(selected_classes)) > 0
    ]
    snapshot_classes = [snapshot_classes[i] for i in keep_index]
    snapshot_classes = [
        sorted(list(set(s_cls) & set(selected_classes))) for s_cls in snapshot_classes
    ]
    return snapshot_classes, keep_index






import re

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
    save_path = os.path.join(save_dir, f"step{step_idx}_frontier{idx}.png")
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
            parts = name.split('_')
            step = int(parts[0].replace('step',''))
            fidx = int(parts[1].replace('frontier',''))
            return (step, fidx)
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
        content.append((" ",))

    # 4. 明确只输出context summary，不要建议
    text = ""
    text += "Please output ONLY a single paragraph context summary, similar to the example above. "
    text += "Do NOT make suggestions or give next-step decisions."
    content.append((text,))

    return sys_prompt, content

def format_frontier_single_prompt(
    question,
    egocentric_imgs,
    frontier_img,
    egocentric_view=False,
    image_goal=None,
    context=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. "
    sys_prompt += "To answer the question, you are required to judge whether this Frontier should be selected to further explore. "
    sys_prompt += "Definitions: "
    sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. "
    sys_prompt += "Selecting a frontier means that you will further explore that direction. "



    if context:
        sys_prompt += "Context: The following summary describes the agent's past exploration and current known status. Use this context to help you make a better choice, but do not treat it as a direct instruction.\n"
        sys_prompt += f"{context}\n"




    content = []
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append((" ",))
    else:
        content.append((text + " ",))

    # egocentric视角可选
    if egocentric_view and len(egocentric_imgs) > 0:
        text = "The following is the egocentric view of the agent in forward direction: "
        content.append((text, egocentric_imgs[-1]))
        content.append((" ",))

    # 只给当前frontier
    text = "Here is the Frontier you need to evaluate: "
    content.append((text, frontier_img))
    content.append((" ",))

    # 0
    text = "Please answer in exactly one of the following two formats:\n"
    text += "Yes\n[State the reason why exploring this frontier is likely to help answer the question]\n"
    text += "or\n"
    text += "No\n[State the reason why exploring this frontier is unlikely to help answer the question]\n"
    text += "Write your answer as a complete sentence focused on whether this frontier could lead to finding the answer, not just describing the current image. "
    text += "Be as proactive as possible: select 'Yes' if there is any meaningful hint that this direction could help answer the question, even if the answer is not immediately obvious. "
    text += "For example:\nYes\nThere is a door in this frontier that may lead to the kitchen, which is relevant to the question.\n"
    text += "or\nNo\nThis frontier only shows a blank wall and does not offer any clue for answering the question.\n"
    text += "Only answer 'Yes' if you believe this frontier is helpful for progressing toward the answer. Otherwise, answer 'No'."

    # 1
    # text = "Please answer in exactly one of the following two formats:\n"
    # text += "Yes\n[Explain why exploring this frontier could help answer the question]\n"
    # text += "or\n"
    # text += "No\n[Explain why exploring this frontier would not help answer the question]\n"
    # text += "Be proactive, but only answer 'Yes' if you see a real possibility to find clues or the answer. Only answer 'No' if you are confident this direction is not helpful at all. "
    # text += "Do not always say 'Yes' or 'No'; decide carefully based on the scene.\n"
    # text += "For example:\nYes\nThere is a door in this frontier that may lead to the kitchen.\n"
    # text += "or\nNo\nThis frontier only shows a blank wall and does not offer any clue for answering the question."


    content.append((text,))

    return sys_prompt, content

from collections import Counter
import random
import re
import logging

def call_openai_api_score(sys_prompt, content, num_trials=5, max_tiebreak_rounds=5, dimension_names=None):
    """
    兼容frontier/listwise/pairwise/yesno单图多维度判定。
    - 如果输出格式为Yes.../No...，就只对那一张图多维度打分。
    - 如果是A.../B...，就分别对两张图打分。
    - 其它情况沿用原有frontier流程。
    - 每次logging.info输出：Frontier X: 4 Yes，或A: 3 Yes/B: 2 Yes
    """
    def extract_imgs(content):
        """
        自动提取所有Frontier图片，顺序排列到extract_imgs列表。
        """
        img_dict = {}
        for c in content:
            if len(c) > 1:
                text = c[0].strip().lower()
                m = re.match(r'frontier\s*(\d+)', text)
                if m:
                    idx = int(m.group(1))
                    img_dict[idx] = c[1]
                elif text.startswith("a"):
                    img_dict['A'] = c[1]
                elif text.startswith("b"):
                    img_dict['B'] = c[1]
        # 按编号升序排列
        # 返回dict，兼容A/B也兼容index
        return img_dict
    def extract_imgs_single(content):
        """
        只抽取('Here is the Frontier you need to evaluate:', <img>)对应的图片。
        返回dict，key为'yes'，value为该图片（找不到则为None）。
        """
        for c in content:
            if len(c) > 1 and isinstance(c[1], str):
                if c[0].strip().lower().startswith('here is the frontier you need to evaluate'):
                    return {'yes': c[1]}
        return {'yes': None}


    def format_frontier_multiscore_prompt(
        question,
        chosen_img,
        dimension_names=None,
        egocentric_img=None,
    ):
        if dimension_names is None:
            dimension_names = [
                "Is exploring this frontier likely to provide information directly related to the question?",
                "Is this frontier likely to reveal new objects or areas relevant for answering the question?",
                "Does this frontier seem to reduce uncertainty about the possible answer?",
                "Will choosing this frontier avoid redundant exploration and save steps?",
                "Is this frontier likely to bring you closer to finding the final answer?"
            ]
        text = ""
        text += "Task: You are an agent exploring an indoor environment to answer a user's question. "
        text += "You have selected a candidate direction (Frontier) to explore next. "
        text += "For this frontier, please answer the following five questions. "
        text += "For each, answer only 'Yes' or 'No' on a single line. "
        text += "Do not add any explanation or extra words.\n"
        text += "Here is an example of the expected output format:\n"
        text += "Yes\nNo\nYes\nNo\nYes\n"
        text += "Now answer the following:\n"

        multi_content = []
        multi_content.append((f"Question: {question}",))
        if egocentric_img is not None:
            multi_content.append(("Current agent egocentric view:", egocentric_img))
            multi_content.append((" ",))
        multi_content.append(("The selected Frontier image:", chosen_img))
        multi_content.append((" ",))
        multi_content.append(("For the selected Frontier, answer each question below with only Yes or No:",))
        for i, dim in enumerate(dimension_names, 1):
            multi_content.append((f"Q{i}: {dim}",))
        multi_content.append(("Write your five answers in five lines, only Yes or No, nothing else.",))
        return text, multi_content

    # --- 自动提取图片（支持编号、A、B） ---
    extract_imgs_dict = extract_imgs(content)
    # 自动提取问题和egocentric视角图
    question = None
    egocentric_img = None
    for c in content:
        if c[0].lower().startswith("question:"):
            question = c[0].replace("Question:", "").strip()
        if "egocentric" in c[0].lower() and len(c) > 1:
            egocentric_img = c[1]

    # 先做一轮API调用，看看输出格式是什么
    resp = call_openai_api(sys_prompt, content)
    if resp is None:
        logging.warning("[Score] API response is None.")
        return None

    resp_str = resp.strip().lower()

    # 1. 只返回yes/no的情况（单张图）
    if resp_str.startswith("yes") or resp_str.startswith("no"):
        if resp_str.startswith("yes"):
            extract_imgs_dict_single = extract_imgs_single(content)
            if extract_imgs_dict_single:
                chosen_img = list(extract_imgs_dict_single.values())[0]
                sys_prompt_score, content_score = format_frontier_multiscore_prompt(
                    question, chosen_img, dimension_names=dimension_names, egocentric_img=egocentric_img
                )
                score_resp = call_openai_api(sys_prompt_score, content_score)
                yes_count = len(re.findall(r"\byes\b", score_resp.strip().lower())) if score_resp else 0
                logging.info(f"[Score] Single Image: {yes_count} Yes")
                if resp_str.startswith("yes") and yes_count < 3:
                    # 替换首个Yes为No
                    resp_new = re.sub(r"^yes", "No", resp, flags=re.IGNORECASE)
                    logging.info("[Score] Changed output from Yes to No due to low multi-score")
                    return resp_new
            return resp
        else:
            # 如果是No，直接返回
            logging.info("[Score] Single Image: No")
            return resp

    # 2. Pairwise（A/B）输出
    elif resp_str.startswith("a") or resp_str.startswith("b"):
        m = re.match(r"([ab])\b", resp_str)
        if m:
            label = m.group(1).upper()
            other_label = 'B' if label == 'A' else 'A'
            chosen_img = extract_imgs_dict.get(label, None)
            if chosen_img is not None:
                sys_prompt_score, content_score = format_frontier_multiscore_prompt(
                    question, chosen_img, dimension_names=dimension_names, egocentric_img=egocentric_img
                )
                score_resp = call_openai_api(sys_prompt_score, content_score)
                yes_count = len(re.findall(r"\byes\b", score_resp.strip().lower())) if score_resp else 0
                logging.info(f"[Score] {label}: {yes_count} Yes")
                # 如果Yes不超过2个，再检测另一个
                if yes_count < 3 and other_label in extract_imgs_dict:
                    other_img = extract_imgs_dict[other_label]
                    sys_prompt_score2, content_score2 = format_frontier_multiscore_prompt(
                        question, other_img, dimension_names=dimension_names, egocentric_img=egocentric_img
                    )
                    score_resp2 = call_openai_api(sys_prompt_score2, content_score2)
                    yes_count2 = len(re.findall(r"\byes\b", score_resp2.strip().lower())) if score_resp2 else 0
                    logging.info(f"[Score] {other_label}: {yes_count2} Yes")
                    # 如果另一项通过，则直接选另一项
                    if yes_count2 >= 3:
                        logging.info(f"[Pairwise] Switch to {other_label} (score passed, {yes_count2} Yes)")
                        # 这里需要返回格式为"{other_label} ..."，但通常LLM只返回你最初的resp
                        # 你可以用正则把reason换过去，也可以直接造一个简单返回
                        return f"{other_label} (switched by score) "
                # 否则返回LLM最初选择的
            return resp


    # 3. 标准frontier投票形式
    else:
        tiebreak_round = 0
        candidate_indices = None
        tried_indices = set()
        last_response = None

        # 抽取所有图片index（按编号排序）
        all_indices = sorted([k for k in extract_imgs_dict if isinstance(k, int)])
        while tiebreak_round < max_tiebreak_rounds:
            responses = []
            indices = []
            for _ in range(num_trials):
                resp = call_openai_api(sys_prompt, content)
                if resp is not None:
                    m = re.match(r"frontier\s+(\d+)", resp.lower())
                    if m:
                        idx = int(m.group(1))
                        if (candidate_indices is None or idx in candidate_indices) and idx not in tried_indices:
                            responses.append(resp)
                            indices.append(idx)
            if not responses:
                logging.warning("[Frontier Score] All responses are None. Return None.")
                return None

            for resp, idx in zip(responses, indices):
                if idx not in extract_imgs_dict:
                    continue
                chosen_img = extract_imgs_dict[idx]
                sys_prompt_score, content_score = format_frontier_multiscore_prompt(
                    question, chosen_img, dimension_names=dimension_names, egocentric_img=egocentric_img
                )
                score_resp = call_openai_api(sys_prompt_score, content_score)
                yes_count = len(re.findall(r"\byes\b", score_resp.strip().lower())) if score_resp else 0
                logging.info(f"[Score] Frontier {idx}: {yes_count} Yes")
                if yes_count >= 3:
                    logging.info(f"[Frontier Score] Selected: frontier {idx} (score passed)")
                    return resp
                tried_indices.add(idx)

            candidate_indices = [idx for idx in set(indices) if idx not in tried_indices]
            tiebreak_round += 1

        if responses:
            logging.info("[Frontier Score] All rounds failed. Randomly returning last tried.")
            return responses[-1]
        else:
            logging.warning("[Frontier Score] No response available at all.")
            return None


def call_openai_api_vote(sys_prompt, content, num_trials=5, max_tiebreak_rounds=5):
    """
    只用于 'yes/no' 投票。返回得票最多的完整回答（开头为Yes或No）。
    日志只记录yes/no计数和最终选择。
    """
    tiebreak_round = 0
    candidate_types = None  # yes or no
    yes_no_pattern = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)
    while True:
        responses = []
        yes_no_list = []
        for _ in range(num_trials):
            resp = call_openai_api(sys_prompt, content)
            if resp is not None:
                resp = resp.strip()
                m = yes_no_pattern.match(resp)
                if m:
                    vote_type = m.group(1).capitalize()
                    if candidate_types is None or vote_type in candidate_types:
                        responses.append(resp)
                        yes_no_list.append(vote_type)
        if not responses:
            logging.warning("[Yes/No Voting] All responses are None. Return None.")
            return None
        # 只统计 Yes/No 数量
        index_counter = Counter(yes_no_list)
        log_str = " | ".join([f"{k}: {v}" for k, v in index_counter.items()])
        logging.info(f"[Yes/No Voting][Round {tiebreak_round+1}] {log_str}")
        max_count = max(index_counter.values())
        winners = [k for k, v in index_counter.items() if v == max_count]
        if len(winners) == 1:
            chosen = winners[0]
            logging.info(f"[Yes/No Voting] Selected: {chosen}")
            # 返回第一个对应类型的完整回答
            for resp in responses:
                m = yes_no_pattern.match(resp)
                if m and m.group(1).capitalize() == chosen:
                    return resp
        else:
            candidate_types = winners
            tiebreak_round += 1
            if tiebreak_round >= max_tiebreak_rounds:
                chosen = random.choice(winners)
                logging.info(f"[Yes/No Voting] Max tie-break rounds reached. Randomly selected: {chosen}")
                for resp in responses:
                    m = yes_no_pattern.match(resp)
                    if m and m.group(1).capitalize() == chosen:
                        return resp

def explore_step(step, cfg, verbose=False, chosen_frontier_path=None, step_idx=None):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        snapshot_id_mapping,
    ) = get_step_info(step, verbose)

    # ==== Step 1: snapshot prompt ====
    sys_prompt, content = format_explore_prompt_snapshot(
        question,
        egocentric_imgs,
        frontier_imgs,  # 可以为空
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
    )

    if verbose:
        logging.info(f"Input prompt (snapshot):")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    retry_bound = 3
    for _ in range(retry_bound):
        full_response = call_openai_api(sys_prompt, content)
        if full_response is None:
            print("call_openai_api (snapshot) returns None, retrying")
            continue

        if isinstance(full_response, list):
            full_response = " ".join(full_response)
        full_response = full_response.strip().lower()

        # snapshot合规判定
        if full_response.startswith("snapshot"):
            tokens = full_response.split()
            if len(tokens) >= 2 and tokens[1].isdigit():
                idx = int(tokens[1])
                if 0 <= idx < len(snapshot_imgs):
                    response = f"{tokens[0]} {tokens[1]}"
                    reason = " ".join(tokens[2:]).strip()
                    reason = clean_reason(reason)  
                    return response, snapshot_id_mapping, reason, len(snapshot_imgs)
                else:
                    print(f"Snapshot index out of range: {tokens[1]}")
                    continue
        elif "no snapshot is available" in full_response:
            # 明确拒绝，直接进入frontier
            break
        else:
            print(f"Unrecognized snapshot response: {full_response}")
            continue

    # ==== Step 2: frontier prompt ====


    context = ''

    if not os.path.exists(chosen_frontier_path):
        os.makedirs(chosen_frontier_path, exist_ok=True)

    png_files = [f for f in os.listdir(chosen_frontier_path) if f.endswith('.png')]
    if len(png_files) > 0:
        sys_prompt_context, content_context = frontier_context(chosen_frontier_path)
        context = call_openai_api(sys_prompt_context, content_context)
    else:
        pass


    # ==== Step 2: frontier prompt ====
    for i, frontier_img in enumerate(frontier_imgs):
        sys_prompt, content = format_frontier_single_prompt(
            question,
            egocentric_imgs,
            frontier_img,
            egocentric_view=step.get("use_egocentric_views", False),
            image_goal=image_goal,
            context=context,
        )
        if verbose:
            logging.info(f"Input prompt (single frontier {i}):")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)
        for attempt in range(retry_bound):
            # full_response = call_openai_api(sys_prompt, content)
            # full_response = call_openai_api_vote(sys_prompt, content)
            full_response = call_openai_api_score(sys_prompt, content)
            if full_response is None:
                print("call_openai_api (single frontier) returns None, retrying")
                continue
            resp = full_response.strip().lower()
            lines = [line.strip() for line in resp.split('\n') if line.strip()]
            first_line = lines[0] if len(lines) > 0 else ""
            if first_line == "yes":
                reason = " ".join(lines[1:]) if len(lines) > 1 else ""
                reason = clean_reason(reason)
                save_base64_to_png(frontier_imgs[i], chosen_frontier_path, step_idx, i)
                return f"frontier {i}", snapshot_id_mapping, reason, len(snapshot_imgs)
            elif first_line == "no":
                reason = " ".join(lines[1:]) if len(lines) > 1 else ""
                if verbose:
                    logging.info(f"frontier {i} -> No (attempt {attempt+1}), reason: {reason}")
                break
            else:
                if verbose:
                    logging.warning(f"frontier {i} unrecognized response: '{first_line}', retrying...")
                continue


    # ==== Step 2b: 如果都没选出来，再用整体frontier prompt兜底 ====
    sys_prompt, content = format_explore_prompt_frontier(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
        context=context,
    )

    if verbose:
        logging.info(f"Input prompt (frontier ALL):")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)

    for _ in range(retry_bound):
        # full_response = call_openai_api(sys_prompt, content)
        full_response = call_openai_api_score(sys_prompt, content)
        if full_response is None:
            print("call_openai_api (frontier ALL) returns None, retrying")
            continue

        if isinstance(full_response, list):
            full_response = " ".join(full_response)
        full_response = full_response.strip().lower()

        if full_response.startswith("frontier"):
            tokens = full_response.split()
            if len(tokens) >= 2 and tokens[1].isdigit():
                idx = int(tokens[1])
                if 0 <= idx < len(frontier_imgs):
                    response = f"{tokens[0]} {tokens[1]}"
                    reason = " ".join(tokens[2:]).strip()
                    reason = clean_reason(reason)
                    save_base64_to_png(frontier_imgs[int(tokens[1])], chosen_frontier_path, step_idx, int(tokens[1]))
                    return response, snapshot_id_mapping, reason, len(snapshot_imgs)
                else:
                    print(f"Frontier index out of range: {tokens[1]}")
                    continue
        else:
            print(f"Unrecognized frontier response: {full_response}")
            continue

    # 如果兜底也失败，返回None
    return None, snapshot_id_mapping, None, len(snapshot_imgs)