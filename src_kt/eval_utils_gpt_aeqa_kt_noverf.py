import openai
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO
import os
import time
from typing import Optional
import logging
from src_kt.const import *
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


# version_1 
# def evaluate_snapshot_relevance_with_full_prompt(
#     vlm, snapshot_img_base64, snapshot_classes, question, tokens=["Yes", "No"], T=1.0
# ):
#     snapshot_img = Image.open(BytesIO(base64.b64decode(snapshot_img_base64))).convert("RGB")

#     # transform = transforms.Compose([
#     #     transforms.Resize((224, 224)),  # 依据你模型默认输入尺寸调整
#     #     transforms.ToTensor()
#     # ])
#     # snapshot_tensor = transform(snapshot_img).unsqueeze(0)

#     class_info = ", ".join(snapshot_classes)
#     sys_prompt = (
#         "You are an intelligent agent in a 3D indoor environment.\n"
#         "You are given a question and a snapshot image that contains the following detected objects:\n"
#         f"{class_info}\n\n"
#         "Determine if the snapshot contains enough information to confidently answer the question.\n"
#     )
#     prompt = f"Question: {question}\nCan you confidently answer the question based on this view? Answer with Yes or No."
#     # reasoning steps in details
#     # different threshold
#     # 2steps verification
#     probs = vlm.get_loss(image=snapshot_img, prompt=sys_prompt + prompt, tokens=tokens, get_smx=True, T=T)


#     return probs




# version_2
# def evaluate_snapshot_relevance_with_full_prompt(
#     vlm, snapshot_img_base64, snapshot_classes, question, tokens=["Yes", "No"], T=1.0
# ):


#     snapshot_img = Image.open(BytesIO(base64.b64decode(snapshot_img_base64))).convert("RGB")
#     class_info = ", ".join(snapshot_classes)

#     # System Prompt —— adding refusal and reasoning steps
#     sys_prompt = (
#         "You are a visual agent operating in a 3D indoor environment.\n"
#         "You are given a natural language question and a snapshot image with the following detected objects:\n"
#         f"{class_info}\n\n"
#         "Your task is to decide whether this snapshot contains enough visual information to confidently answer the question.\n"
#         "If the snapshot is sufficient and you are confident, respond with 'Yes'.\n"
#         "If the snapshot is insufficient or uncertain, respond with 'No'. Do not guess.\n"
#         "Only answer based on the visual evidence in the snapshot. Do not infer from prior knowledge.\n"
#     )

#     prompt = (
#         f"Question: {question}\n"
#         "Step 1: Check the listed objects.\n"
#         "Step 2: Consider whether they provide enough evidence to answer the question.\n"
#         "Step 3: Respond only with one word: Yes or No."
#     )

#     # Call VLM for loss-based softmax scoring
#     probs = vlm.get_loss(
#         image=snapshot_img,
#         prompt=sys_prompt + prompt,
#         tokens=tokens,
#         get_smx=True,
#         T=T,
#     )

#     return probs





#version_3
# def evaluate_snapshot_relevance_with_full_prompt(
#     vlm, snapshot_img_base64, snapshot_classes, question, tokens=["Yes", "No"], T=1.0
# ):


#     few_shot_examples = """
#         [Examples]

#         Question: Is there a laptop on the desk?
#         Detected objects: desk, chair, lamp, laptop
#         Are you confident you can answer the question based solely on this snapshot?
#         Answer: Yes

#         ---

#         Question: Is there a red mug on the dining table?
#         Detected objects: dining table, chair, mug
#         Are you confident you can answer the question based solely on this snapshot?
#         Answer: No

#         ---

#         Question: Is there enough space to walk between the sofa and the TV stand?
#         Detected objects: sofa, TV stand, coffee table, rug
#         Are you confident you can answer the question based solely on this snapshot?
#         Answer: Yes

#         ---

#         Question: What is written on the whiteboard?
#         Detected objects: whiteboard, marker, eraser
#         Are you confident you can answer the question based solely on this snapshot?
#         Answer: No

#         ---

#         Question: Can you see the microwave in the kitchen area?
#         Detected objects: countertop, sink, oven
#         Are you confident you can answer the question based solely on this snapshot?
#         Answer: No

#         ---
#         """

#     snapshot_img = Image.open(BytesIO(base64.b64decode(snapshot_img_base64))).convert("RGB")
#     class_info = ", ".join(snapshot_classes)

#     prompt = (
#         "You are a visual agent in a 3D environment. For each question, you are given a snapshot image and a list of detected objects in that image.\n"
#         "Your task is to answer only this: \"Are you confident you can answer the question based solely on this snapshot?\"\n"
#         "Answer with \"Yes\" if you are confident that the snapshot contains enough visual evidence. Otherwise, answer \"No\".\n"
#         "Do not guess. Only answer \"Yes\" if you are truly confident.\n"
#         + few_shot_examples +
#         f"\nQuestion: {question}\nDetected objects: {class_info}\nAre you confident you can answer the question based solely on this snapshot?\nAnswer:"
#     )

#     # 3. 调用VLM
#     probs = vlm.get_loss(
#         image=snapshot_img,
#         prompt=prompt,
#         tokens=tokens,
#         get_smx=True,
#         T=T,
#     )


#     few_shot_anwer = """
#     [Examples]

#     Question: What is the white object on the wall above the TV?
#     Detected objects: desk, chair, lamp, laptop, Air conditioning unit, Air conditioner
#     Reasoning: The detected objects include an air conditioning unit and the question asks about a white object above the TV, which matches the position of an air conditioner in typical rooms.
#     Answer: Air conditioning unit

#     ---

#     Question: Is there a red mug on the dining table?
#     Detected objects: dining table, chair, mug
#     Reasoning: There is a mug detected, but no color information can be clearly seen from the snapshot, so I cannot be confident about its color.
#     Answer: No
#     """

#     prompt_answer = (
#         "You are a visual agent in a 3D environment. For each question, you are given a snapshot image and a list of detected objects in that image.\n"
#         "Your task is to think step by step: first provide a brief reasoning process based solely on the visual information in the snapshot, then give the final answer.\n"
#         + few_shot_anwer +
#         f"\nQuestion: {question}\nDetected objects: {class_info}\nReasoning: <your reasoning>\nAnswer: <your answer>\n"
#     )

#     # 调用生成接口
#     reasoning_and_answer = vlm.generate(
#         image=snapshot_img,
#         prompt=prompt_answer,
#         T=T,
#     )


#     few_shot_verification = """
#         [Examples]

#         Question: What is the white object on the wall above the TV?
#         Detected objects: desk, chair, lamp, laptop, Air conditioning unit, Air conditioner
#         Answer: Air conditioning unit
#         Does this answer confidently and correctly answer the question based on the snapshot?
#         Yes

#         ---

#         Question: Is there a red mug on the dining table?
#         Detected objects: dining table, chair, mug
#         Answer: No
#         Does this answer confidently and correctly answer the question based on the snapshot?
#         Yes

#         ---

#         Question: What is written on the whiteboard?
#         Detected objects: whiteboard, marker, eraser
#         Answer: Meeting agenda
#         Does this answer confidently and correctly answer the question based on the snapshot?
#         No

#         ---

#         Question: Is there enough space to walk between the sofa and the TV stand?
#         Detected objects: sofa, TV stand, coffee table, rug
#         Answer: Yes
#         Does this answer confidently and correctly answer the question based on the snapshot?
#         Yes
#         """


#     prompt_verification = (
#     "You are a visual agent in a 3D environment. For each question, you are given a snapshot image, a list of detected objects, and an answer.\n"
#     "Please decide: Does this answer confidently and correctly answer the question based on the visual evidence in the snapshot?\n"
#     + few_shot_verification +
#     f"\nQuestion: {question}\nDetected objects: {class_info}\nAnswer: {reasoning_and_answer}\nDoes this answer confidently and correctly answer the question based on the snapshot?")

#     probs_verify = vlm.get_loss(
#         image=snapshot_img,
#         prompt=prompt_verification,
#         tokens=tokens,
#         get_smx=True,
#         T=T,
#     )

#     turn_snapshot = False
#     if probs_verify[0] > 0.8:
#         turn_snapshot = True
#         print(f"Snapshot is selected with probability: {probs_verify[0]}")



    

#     return probs, turn_snapshot







# version_4
def evaluate_snapshot_relevance_with_full_prompt(
    vlm, snapshot_img_base64, snapshot_classes, question, tokens=["Yes", "No"], T=1.0
):


    few_shot_examples = """
        [Examples]

        Question: Is there a laptop on the desk?
        Detected objects: desk, chair, lamp, laptop
        Are you confident you can answer the question based solely on this snapshot?
        Answer: Yes

        ---

        Question: Is there a red mug on the dining table?
        Detected objects: dining table, chair, mug
        Are you confident you can answer the question based solely on this snapshot?
        Answer: No

        ---

        Question: Is there enough space to walk between the sofa and the TV stand?
        Detected objects: sofa, TV stand, coffee table, rug
        Are you confident you can answer the question based solely on this snapshot?
        Answer: Yes

        ---

        Question: What is written on the whiteboard?
        Detected objects: whiteboard, marker, eraser
        Are you confident you can answer the question based solely on this snapshot?
        Answer: No

        ---

        Question: Can you see the microwave in the kitchen area?
        Detected objects: countertop, sink, oven
        Are you confident you can answer the question based solely on this snapshot?
        Answer: No

        ---
        """

    snapshot_img = Image.open(BytesIO(base64.b64decode(snapshot_img_base64))).convert("RGB")
    class_info = ", ".join(snapshot_classes)

    prompt = (
        "You are a visual agent in a 3D environment. For each question, you are given a snapshot image and a list of detected objects in that image.\n"
        "Your task is to answer only this: \"Are you confident you can answer the question based solely on this snapshot?\"\n"
        "Answer with \"Yes\" if you are confident that the snapshot contains enough visual evidence. Otherwise, answer \"No\".\n"
        "Do not guess. Only answer \"Yes\" if you are truly confident.\n"
        + few_shot_examples +
        f"\nQuestion: {question}\nDetected objects: {class_info}\nAre you confident you can answer the question based solely on this snapshot?\nAnswer:"
    )

    # 3. 调用VLM
    probs = vlm.get_loss(
        image=snapshot_img,
        prompt=prompt,
        tokens=tokens,
        get_smx=True,
        T=T,
    )


    few_shot_anwer = """
    [Examples]

    Question: What is the white object on the wall above the TV?
    Detected objects: desk, chair, lamp, laptop, Air conditioning unit, Air conditioner
    Reasoning: The detected objects include an air conditioning unit and the question asks about a white object above the TV, which matches the position of an air conditioner in typical rooms.
    Answer: Air conditioning unit

    ---

    Question: Is there a red mug on the dining table?
    Detected objects: dining table, chair, mug
    Reasoning: There is a mug detected, but no color information can be clearly seen from the snapshot, so I cannot be confident about its color.
    Answer: No
    """

    prompt_answer = (
        "You are a visual agent in a 3D environment. For each question, you are given a snapshot image and a list of detected objects in that image.\n"
        "Your task is to think step by step: first provide a brief reasoning process based solely on the visual information in the snapshot, then give the final answer.\n"
        + few_shot_anwer +
        f"\nQuestion: {question}\nDetected objects: {class_info}\nReasoning: <your reasoning>\nAnswer: <your answer>\n"
    )

    # 调用生成接口
    reasoning_and_answer = vlm.generate(
        image=snapshot_img,
        prompt=prompt_answer,
        T=T,
    )


    few_shot_verification = """
        [Examples]

        Question: What is the white object on the wall above the TV?
        Detected objects: desk, chair, lamp, laptop, Air conditioning unit, Air conditioner
        Answer: Air conditioning unit
        Does this answer confidently and correctly answer the question based on the snapshot?
        Yes

        ---

        Question: Is there a red mug on the dining table?
        Detected objects: dining table, chair, mug
        Answer: No
        Does this answer confidently and correctly answer the question based on the snapshot?
        Yes

        ---

        Question: What is written on the whiteboard?
        Detected objects: whiteboard, marker, eraser
        Answer: Meeting agenda
        Does this answer confidently and correctly answer the question based on the snapshot?
        No

        ---

        Question: Is there enough space to walk between the sofa and the TV stand?
        Detected objects: sofa, TV stand, coffee table, rug
        Answer: Yes
        Does this answer confidently and correctly answer the question based on the snapshot?
        Yes
        """


    prompt_verification = (
    "You are a visual agent in a 3D environment. For each question, you are given a snapshot image, a list of detected objects, and an answer.\n"
    "Please decide: Does this answer confidently and correctly answer the question based on the visual evidence in the snapshot?\n"
    + few_shot_verification +
    f"\nQuestion: {question}\nDetected objects: {class_info}\nAnswer: {reasoning_and_answer}\nDoes this answer confidently and correctly answer the question based on the snapshot?")

    probs_verify = vlm.get_loss(
        image=snapshot_img,
        prompt=prompt_verification,
        tokens=tokens,
        get_smx=True,
        T=T,
    )

    turn_snapshot = False
    if probs_verify[0] > 0.8:
        turn_snapshot = True
        print(f"Snapshot is selected with probability: {probs_verify[0]}")



    

    return probs, turn_snapshot








from collections import defaultdict
def pairwise_voting_frontier_list(frontier_b64_list, question, selector):
    """
    Round-robin voting among base64-encoded frontier images.
    自动为所有frontier批量生成caption，pairwise对比时直接用，不重复生成。
    输入：
      - frontier_b64_list: list of base64-encoded images
      - question: string
      - selector: LLaVAFrontierSelector 实例
    输出：
      - winner_idx (int), decisive_reason (str)
      （与你的原版接口完全一致）
    """
    import base64
    from io import BytesIO

    n = len(frontier_b64_list)
    if n == 0:
        print("No frontiers to compare.", flush=True)
        return -1, "", []
    if n == 1:
        print("Only one frontier provided: automatically selected as winner.", flush=True)
        return 0, "Only one frontier provided: automatically selected as winner.", []

    # 1. 先批量解码为numpy array
    decoded_imgs = []
    for idx, b64 in enumerate(frontier_b64_list):
        pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        decoded_imgs.append(np.array(pil))

    # 2. 先批量生成所有frontier的caption
    print("[INFO] Generating captions for all frontiers ...", flush=True)
    captions = {}
    for idx, img in enumerate(decoded_imgs):
        captions[idx] = selector.generate_caption(img)
        print(f"Frontier {idx}: {captions[idx]}", flush=True)
    print("[INFO] Caption generation completed.\n", flush=True)

    # 3. 两两比较时直接用caption，不重复生成
    scores = {i: 0 for i in range(n)}
    win_reasons = defaultdict(list)
    log = []

    for i in range(n):
        for j in range(i + 1, n):
            img_i, img_j = decoded_imgs[i], decoded_imgs[j]
            cap_i, cap_j = captions[i], captions[j]

            print(f"[Compare] Frontier {i} (A) vs Frontier {j} (B)", flush=True)
            win1, ans1, reason1, _ = selector.select(cap_i, img_i, cap_j, img_j, question)
            log.append((i, j, 'A', win1, ans1, reason1))
            if win1 == 0:
                print(f" - Round 1 winner: Frontier {i}. Reason: {reason1}", flush=True)
            elif win1 == 1:
                print(f" - Round 1 winner: Frontier {j}. Reason: {reason1}", flush=True)
            else:
                print(" - Round 1: No clear winner.", flush=True)

            print(f"[Compare] Frontier {j} (A) vs Frontier {i} (B)", flush=True)
            win2, ans2, reason2, _ = selector.select(cap_j, img_j, cap_i, img_i, question)
            log.append((j, i, 'A', win2, ans2, reason2))
            if win2 == 0:
                print(f" - Round 2 winner: Frontier {j}. Reason: {reason2}", flush=True)
            elif win2 == 1:
                print(f" - Round 2 winner: Frontier {i}. Reason: {reason2}", flush=True)
            else:
                print(" - Round 2: No clear winner.", flush=True)

            # Scoring
            if win1 == 0 and win2 == 1:
                scores[i] += 3
                win_reasons[i].append(reason1)
                print(f" --> Frontier {i} wins both rounds: +3 points.", flush=True)
            elif win1 == 1 and win2 == 0:
                scores[j] += 3
                win_reasons[j].append(reason1)
                print(f" --> Frontier {j} wins both rounds: +3 points.", flush=True)
            else:
                scores[i] += 1
                scores[j] += 1
                print(f" --> Split result: Frontier {i} +1, Frontier {j} +1.", flush=True)

    print("\n=== Final voting results ===", flush=True)
    for idx in range(n):
        print(f"Frontier {idx}: score = {scores[idx]}", flush=True)

    # Determine winner
    max_score = max(scores.values())
    top_indices = [idx for idx, s in scores.items() if s == max_score]

    if len(top_indices) == 1:
        winner = top_indices[0]
        reason = win_reasons[winner][0] if win_reasons[winner] else ""
        print(f"Winner: Frontier {winner}. Decisive reason: {reason}", flush=True)
        return winner, reason, top_indices
    else:
        print("Tie or no clear winner. Top candidates:", top_indices, flush=True)
        reason = "no clear winner, only do some selection"
        return -1, reason, top_indices


def tournament_voting_frontier_list(frontier_b64_list, question, selector, max_rounds=3):
    """
    Tournament (knockout) voting among base64-encoded frontier images.
    Each round, frontiers are paired and compared in both directions, winner advances, until only one remains.
    If after max_rounds there are still multiple candidates (all ties), randomly select one.
    """
    import base64
    from io import BytesIO
    import numpy as np
    from PIL import Image
    import random

    n = len(frontier_b64_list)
    if n == 0:
        print("No frontiers to compare.", flush=True)
        return -1, ""
    if n == 1:
        print("Only one frontier provided: automatically selected as winner.", flush=True)
        return 0, "Only one frontier provided: automatically selected as winner."

    # 1. Decode all images
    decoded_imgs = []
    for idx, b64 in enumerate(frontier_b64_list):
        pil = Image.open(BytesIO(base64.b64decode(b64))).convert("RGB")
        decoded_imgs.append(np.array(pil))

    # 2. Generate all captions (if needed)
    print("[INFO] Generating captions for all frontiers ...", flush=True)
    captions = {}
    for idx, img in enumerate(decoded_imgs):
        captions[idx] = selector.generate_caption(img)
        print(f"Frontier {idx}: {captions[idx]}", flush=True)
    print("[INFO] Caption generation completed.\n", flush=True)

    current_indices = list(range(n))
    round_num = 1
    last_reason = ""
    while len(current_indices) > 1 and round_num <= max_rounds:
        print(f"\n=== Tournament Round {round_num}, {len(current_indices)} candidates ===", flush=True)
        next_round_indices = []
        round_reasons = {}
        i = 0
        while i < len(current_indices):
            if i == len(current_indices) - 1:
                # Odd one advances automatically
                solo = current_indices[i]
                print(f"Frontier {solo} advances by default (odd number).", flush=True)
                next_round_indices.append(solo)
                i += 1
                continue
            a = current_indices[i]
            b = current_indices[i + 1]
            print(f"[Compare] Frontier {a} (A) vs Frontier {b} (B)", flush=True)
            win1, ans1, reason1, _ = selector.select(captions[a], decoded_imgs[a], captions[b], decoded_imgs[b], question)
            win2, ans2, reason2, _ = selector.select(captions[b], decoded_imgs[b], captions[a], decoded_imgs[a], question)
            print(f" - [A vs B] winner: {'A' if win1==0 else ('B' if win1==1 else 'Tie')}, reason: {reason1}")
            print(f" - [B vs A] winner: {'A' if win2==1 else ('B' if win2==0 else 'Tie')}, reason: {reason2}")

            if win1 == 0 and win2 != 0:
                print(f" --> Frontier {a} wins both directions and advances.")
                next_round_indices.append(a)
                round_reasons[a] = reason1
                last_reason = reason1
            elif win1 == 1 and win2 != 1:
                print(f" --> Frontier {b} wins both directions and advances.")
                next_round_indices.append(b)
                round_reasons[b] = reason1
                last_reason = reason1
            else:
                print(f" --> No clear winner, both advance.")
                next_round_indices.append(a)
                next_round_indices.append(b)
                # Tie reason优先A的reason
                round_reasons[a] = "Tie, both advance"
                round_reasons[b] = "Tie, both advance"
            i += 2
        # 去重（极端全平局时防止死循环）
        current_indices = list(dict.fromkeys(next_round_indices))
        round_num += 1

    # 判断是否最终有唯一winner
    if len(current_indices) == 1:
        winner = current_indices[0]
        reason = round_reasons.get(winner, last_reason)
        print(f"Final winner: Frontier {winner}. Reason: {reason}", flush=True)
        return winner, reason
    else:
        print(f"After {max_rounds} rounds, still tie. Randomly pick one.", flush=True)
        winner = random.choice(current_indices)
        return winner, f"All tied after {max_rounds} rounds, randomly selected."



















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
        "Now, you should give a final answer to the question **based on this snapshot only**.\n"
        "Instructions:\n"
        "- Your answer should be a direct, natural sentence that a human can understand.\n"
        "- DO NOT mention words like 'snapshot', 'in the image', 'on the left', or any reference to image layout.\n"
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
        "Only return the answer you generated, nothing else."))

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


def explore_step(threshold, llava_pairwise_selector, vlm, step, cfg, verbose=False):
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


    # snapshot_probs = []
    # turn_snapshot_list = []
    # for snapshot_img_base64, classes in zip(snapshot_imgs, snapshot_classes):
    #     # vlm.model.llm_backbone.half_precision_dtype = torch.float16
    #     prob, turn_snapshot = evaluate_snapshot_relevance_with_full_prompt(vlm, snapshot_img_base64, classes, question)
    #     snapshot_probs.append(prob)
    #     turn_snapshot_list.append(turn_snapshot)

    # # only select snapshots with a probability of yes above the threshold
    # qualified_indices = [i for i, probs in enumerate(snapshot_probs) if probs[0] > threshold and turn_snapshot_list[i]]

    # if qualified_indices:
    #     best_index = max(qualified_indices, key=lambda i: snapshot_probs[i][0] - snapshot_probs[i][1])
    #     final_response = f"snapshot {best_index}"

    #     sys_prompt_explain, content_explain = format_explore_prompt_end(
    #         question=question,
    #         snapshot_img=snapshot_imgs[best_index],
    #         snapshot_classes=snapshot_classes[best_index],
    #         image_goal=image_goal,
    #     )
    #     final_reason = call_openai_api(sys_prompt_explain, content_explain)
    #     if final_reason is None:
    #         final_reason = "No explanation provided."





    # else:

        

    if verbose:
        logging.info(f"Input prompt:")
        message = sys_prompt
        for c in content:
            message += c[0]
            if len(c) == 2:
                message += f"[{c[1][:10]}...]"
        logging.info(message)
    
    frontier_index, reason = tournament_voting_frontier_list(frontier_imgs, question, llava_pairwise_selector)
    print(f"Frontier voting result: {frontier_index}, reason: {reason}", flush=True)
    

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
            if choice_type == "frontier":
                # mapped_idx = top_indices[int(choice_id)]
                final_response = f"frontier {frontier_index}"
                print("choice id:", int(choice_id), flush=True)
                print("choose the index by llava knockout tournament:", frontier_index, flush=True)
            else:
                final_response = response
            final_reason = reason
            break
    

    return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs)
