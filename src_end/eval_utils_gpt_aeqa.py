import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from src_end.const import *
import torch
import argparse
from omegaconf import OmegaConf
import random
import numpy as np
import torch
import time
import json
import logging
import matplotlib.pyplot as plt



from dotenv import load_dotenv
load_dotenv(dotenv_path="/home/wiss/zhang/code/openeqa/3D-Mem/.env", override=True)
client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


from PIL import Image
from io import BytesIO
import base64
import torch
from torchvision import transforms

def evaluate_snapshot_relevance_with_full_prompt(
    vlm, snapshot_img_base64, snapshot_classes, question, tokens=["Yes", "No"], T=1.0
):
    # 将 base64 解码为 PIL 图片
    snapshot_img = Image.open(BytesIO(base64.b64decode(snapshot_img_base64))).convert("RGB")

    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),  # 依据你模型默认输入尺寸调整
    #     transforms.ToTensor()
    # ])
    # snapshot_tensor = transform(snapshot_img).unsqueeze(0)

    # 构造简洁有效的 prompt（只问：这张图是否足够回答问题）
    class_info = ", ".join(snapshot_classes)
    sys_prompt = (
        "You are an intelligent agent in a 3D indoor environment.\n"
        "You are given a question and a snapshot image that contains the following detected objects:\n"
        f"{class_info}\n\n"
        "Determine if the snapshot contains enough information to confidently answer the question.\n"
    )
    prompt = f"Question: {question}\nCan you confidently answer the question based on this view? Answer with Yes or No."

    # 调用 VLM 模型
    probs = vlm.get_loss(image=snapshot_img, prompt=sys_prompt + prompt, tokens=tokens, get_smx=True, T=T)
    if probs[1]>probs[0]:
        turn_snapshot=False
    else:
        turn_snapshot=True

    return probs, turn_snapshot


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
                model="gpt-4o",  # model = "deployment_name"
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
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose either a snapshot as the answer or a Frontier to further explore.\n"
    sys_prompt += "Definitions:\n"
    sys_prompt += "snapshot: A focused observation of several objects. Choosing a snapshot means that this snapshot image contains enough information for you to answer the question. "
    sys_prompt += "If you choose a snapshot, you need to directly give an answer to the question. If you don't have enough information to give an answer, then don't choose a snapshot.\n"
    sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. Selecting a frontier means that you will further explore that direction. "
    sys_prompt += "If you choose a Frontier, you need to explain why you would like to choose that direction to explore.\n"

    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        content.append(("\n",))
    else:
        content.append((text + "\n",))

    text = "Select the Frontier/snapshot that would help find the answer of the question.\n"
    content.append((text,))

    # 2 add egocentric view
    if egocentric_view:
        text = (
            "The following is the egocentric view of the agent in forward direction: "
        )
        content.append((text, egocentric_imgs[-1]))
        content.append(("\n",))

    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can choose (followed with contained object classes)\n"
    text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model. "
    text += "So you still need to utilize the images to make decisions.\n"
    content.append((text,))
    if len(snapshot_imgs) == 0:
        content.append(("No snapshot is available\n",))
    else:
        for i in range(len(snapshot_imgs)):
            content.append((f"snapshot {i} ", snapshot_imgs[i]))
            if use_snapshot_class:
                text = ", ".join(snapshot_classes[i])
                content.append((text,))
            content.append(("\n",))

    # 4 here is the frontier images
    text = "The followings are all the Frontiers that you can explore: \n"
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available\n",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            content.append(("\n",))

    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'snapshot i\n[Answer]' or 'Frontier i\n[Reason]', where i is the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the first snapshot, you can return 'snapshot 0\nThe fruit bowl is on the kitchen counter.'. "
    text += "If you choose the second frontier, you can return 'Frontier 1\nI see a door that may lead to the living room.'.\n"
    text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; "
    text += "(2) you can also utilize other snapshots, frontiers and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question.\n"
    content.append((text,))

    return sys_prompt, content



def format_explore_prompt_end(
    question,
    snapshot_img,          
    snapshot_classes,     
    image_goal=None,
):
    # system prompt: tell the agent to give a final answer based on the snapshot
    sys_prompt = (
        "Task: You are an agent in a 3D indoor environment tasked with answering a question.\n"
        "You have already selected one snapshot image that contains several detected objects.\n"
        "Now, you should give a final answer to the question **based on this snapshot only**.\n\n"
        "Instructions:\n"
        "- Your answer should be a direct, natural sentence that a human can understand.\n"
        "- DO NOT mention words like 'snapshot', 'in the image', 'on the left', or any reference to image layout.\n"
        "- Also provide a short reason why this snapshot helps you answer the question."
    )

    # content to be sent to the model
    content = []

    if image_goal is not None:
        content.append((f"Question: {question}", image_goal))
    else:
        content.append((f"Question: {question}",))

    # Snapshot 
    content.append(("Here is the selected snapshot that may help answer the question:", snapshot_img))

    # Object 
    class_text = ", ".join(snapshot_classes)
    content.append((f"Objects detected in this snapshot: {class_text}",))

    # final answer format
    content.append((
        "Please respond in the following format:\n"
        "'Answer: <your answer>'\n"
        "'Reason: <your reason>'\n"
        "Only return these two lines, nothing else."
    ,))

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


def explore_step(vlm, step, cfg, verbose=False):
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
    sys_prompt, content = format_explore_prompt(
        question,
        egocentric_imgs,
        frontier_imgs,
        snapshot_imgs,
        snapshot_classes,
        egocentric_view=step.get("use_egocentric_views", False),
        use_snapshot_class=True,
        image_goal=image_goal,
    )


    # 计算每个snapshot的概率（完整prompt）
    snapshot_probs = []
    for snapshot_img_base64, classes in zip(snapshot_imgs, snapshot_classes):
        # vlm.model.llm_backbone.half_precision_dtype = torch.float16
        prob, turn_snapshot = evaluate_snapshot_relevance_with_full_prompt(vlm, snapshot_img_base64, classes, question)
        snapshot_probs.append(prob)

    # check turn_snapshot True or False
    turn_snapshot_flags = []
    snapshot_deltas = []

    for probs in snapshot_probs:
        delta = probs[0] - probs[1]  # Yes - No
        snapshot_deltas.append(delta)
        turn_snapshot_flags.append(probs[0] > probs[1])  # True if Yes is more likely than No

    if any(turn_snapshot_flags):
        best_index = int(np.argmax(snapshot_deltas))  
        final_response = f"snapshot {best_index}"  # query_vlm_for_response will return the index of the best snapshot
        # final_reason = "Selected based on highest confidence from evaluate_snapshot_relevance_with_full_prompt"

        sys_prompt_explain, content_explain = format_explore_prompt_end(
            question=question,
            snapshot_img=snapshot_imgs[best_index],
            snapshot_classes=snapshot_classes[best_index],
            image_goal=image_goal,
        )
        final_reason = call_openai_api(sys_prompt_explain, content_explain)
        if final_reason is None:
            final_reason = "No explanation provided."












    else:        # No snapshot is selected first, we need to select a frontier
        if verbose:
            logging.info(f"Input prompt:")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)

        retry_bound = 3
        final_response = None
        final_reason = None
        for _ in range(retry_bound):
            full_response = call_openai_api(sys_prompt, content)

            if full_response is None:
                print("call_openai_api returns None, retrying")
                continue

            full_response = full_response.strip()
            if "\n" in full_response:
                full_response = full_response.split("\n")
                response, reason = full_response[0], full_response[-1]
                response, reason = response.strip(), reason.strip()
            else:
                response = full_response
                reason = ""
            response = response.lower()
            try:
                choice_type, choice_id = response.split(" ")
            except Exception as e:
                print(f"Error in splitting response: {response}")
                print(e)
                continue

            response_valid = False
            if (
                choice_type == "snapshot"
                and choice_id.isdigit()
                and 0 <= int(choice_id) < len(snapshot_imgs)
            ):
                response_valid = True
            elif (
                choice_type == "frontier"
                and choice_id.isdigit()
                and 0 <= int(choice_id) < len(frontier_imgs)
            ):
                response_valid = True

            if response_valid:
                final_response = response
                final_reason = reason
                break

    return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs)
