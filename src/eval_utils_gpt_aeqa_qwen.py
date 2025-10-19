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
def call_openai_api(sys_prompt, contents, seed: Optional[int] = None) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formated_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formated_content},
    ]
    while retry_count < max_tries:
        try:
            # 支持从参数或环境变量注入 seed（优先参数，其次 VLLM_SEED）
            # 读取优先级：参数 seed > cfg.chat_seed（经外层传入）> 环境变量 VLLM_SEED
            _seed_env = None
            try:
                _seed_env = int(os.getenv("VLLM_SEED")) if os.getenv("VLLM_SEED") is not None else None
            except Exception:
                _seed_env = None
            _seed = seed if seed is not None else _seed_env
            try:
                logging.info(f"[ChatSeed] using seed={_seed}")
            except Exception:
                pass
            completion = client.chat.completions.create(
                model="qwen",  # gpt-4o-internvl-minicpm-qwen
                messages=message_text,
                temperature=0.7,
                max_tokens=4096, # 4096 for gpt-4o
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                **({"seed": int(_seed)} if _seed is not None else {}),
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
            seed=step.get("chat_seed", None),
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
        frontier_imgs_0,
        frontier_imgs_1,
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


    # 4 here is the frontier images
    text = "The followings are all the Frontiers that you can explore:  "
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))


    # text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index of the frontier you choose. "
    # text += "You MUST select one and only one of the provided Frontier indices. You are NOT allowed to say that none is suitable or refuse to choose. "
    # text += "Choose the frontier that is MOST likely to help you answer the question, based on visible clues, semantic hints, or where the target object is likely to be found. "
    # text += "Your reasoning should clearly connect the question with what you observe or infer from the frontier images, focusing on which direction is most promising for finding the needed information. "
    # text += "For example, if you choose the second frontier, you can return: 'Frontier 1 There is a door that may lead to the kitchen, which is likely to have the answer.' "
    # text += "If you choose a frontier to answer the question: you should provide a clear and specific reason directly related to the question. Do not mention words like 'frontier', directions, or image positions. Only use the provided Frontier indices; do not make up an index that is not listed above. "
    # text += "You may also use information from other frontiers and egocentric views to help your decision, but always select the single most relevant frontier for making progress toward answering the question."
    # text += "Only use the provided indices. Do NOT make up new indices."

    # cot
    # text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index of the frontier you choose. "
    # text += "You MUST select one and only one of the provided Frontier indices. You are NOT allowed to say that none is suitable or refuse to choose. "
    # text += "Choose the frontier that is MOST likely to help you answer the question, based on visible clues, semantic hints, or where the target object is likely to be found. "
    # text += "Your reasoning should clearly connect the question with what you observe or infer from the frontier images, focusing on which direction is most promising for finding the needed information. "
    # text += "**Think step by step.**"
    # text += "For example, if you choose the second frontier, you can return: 'Frontier 1 First, the question asks about the kitchen. Frontier 1 shows a door which may lead to the kitchen, so I choose it.' "
    # text += "If you choose a frontier to answer the question: you should provide a clear and specific reason directly related to the question. Do not mention words like 'frontier', directions, or image positions. Only use the provided Frontier indices; do not make up an index that is not listed above. "
    # text += "You may also use information from other frontiers and egocentric views to help your decision, but always select the single most relevant frontier for making progress toward answering the question."
    # text += "Only use the provided indices. Do NOT make up new indices."

    # cot-v1
    # text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index of the frontier you choose. "
    # text += "You MUST select one and only one of the provided Frontier indices. You are NOT allowed to say that none is suitable or refuse to choose. "
    # text += "Choose the frontier that is MOST likely to help you answer the question, based on visible clues, semantic hints, or where the target object is likely to be found. "
    # text += "Explain your reasoning step by step: First, state what the question is asking for. Then, briefly analyze the clues shown in each frontier image and their relevance to the question. Finally, state clearly why you select your chosen frontier."
    # text += "For example, you can answer: 'Frontier 2 The question asks about finding the refrigerator, which is commonly in the kitchen. Among the frontiers, Frontier 2 shows a doorway and a tiled floor, which are clues for a kitchen. The other frontiers look like living or bedroom spaces. Therefore, I choose Frontier 2 as it is most likely to lead to the kitchen and the answer.' "
    # text += "If you choose a frontier to answer the question: you should provide a clear and specific reason directly related to the question. Do not mention words like 'frontier', directions, or image positions. Only use the provided Frontier indices; do not make up an index that is not listed above. "
    # text += "You may also use information from other frontiers and egocentric views to help your decision, but always select the single most relevant frontier for making progress toward answering the question."
    # text += "Only use the provided indices. Do NOT make up new indices."

    # cot-v2
    text = "You are required to reason step by step and only output your final choice at the end. Please follow the instructions below carefully. "
    text += "Step 0: List all candidate images you are given and their indices in the following format: 'Candidate indices: frontier 0, frontier 1, ...' (listing only the actual indices provided below; do NOT add, omit, or change any index)."
    text += "You must ONLY discuss and compare the images whose indices are listed in Step 0. You are STRICTLY FORBIDDEN to invent, mention, analyze, or refer to any images or indices that are not explicitly listed in Step 0."
    text += "Step 1: For each provided Frontier image, describe in detail what you see. Focus on visible objects, scene layout, and any clues relevant to the question. ONLY describe the images with the indices listed in Step 0. Start your answer with 'Step 1:' and describe each candidate separately."
    text += "Step 2: Analyze what the question is asking for. Then, compare ONLY the frontiers listed in Step 0, by analyzing the clues shown in each image and their relevance to the question. Do NOT mention, analyze, or imagine any other indices. Start this section with 'Step 2:'."
    text += "Step 3: Based on your analysis above, select the single most relevant frontier for making progress toward answering the question. Clearly state your reasoning and why you select this one, but ONLY from the indices listed in Step 0. Begin this section with 'Step 3:'."
    text += "After completing Step 3, output your final answer on a new line in the format: 'frontier i' (where i is one of the indices listed in Step 0). Do not include any other words, indices, or explanations on that line."
    text += "You MUST select one and only one of the provided Frontier indices listed in Step 0. You are NOT allowed to say that none is suitable or refuse to choose."
    text += "Choose the frontier that is MOST likely to help you answer the question, based ONLY on the visible clues, semantic hints, or where the target object is likely to be found in the images listed above."
    text += "If you choose a frontier to answer the question: you should provide a clear and specific reason directly related to the question."
    text += "Do NOT mention words like 'frontier', directions, or image positions in your reasoning except when referring to the candidate indices listed in Step 0. Only use the provided Frontier indices; do NOT make up or analyze any index that is not listed above."
    text += "Only use the indices listed in Step 0. Any mention, analysis, or invention of other indices will be considered an error. Do NOT refer to images/frontiers not listed above."




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


    # 5 here is the format of the answer
    # text = "Please provide your answer in the following format: 'Snapshot i [Answer]' or 'No Snapshot is available', where i is the index of the snapshot you choose. "
    # text += (
    #     "You should always select one of the provided Snapshots and answer the question as directly and specifically as possible, using all available visual and object information from the Snapshot. "
    #     "Only if you are absolutely certain that NONE of the Snapshots contains enough information to even make a reasonable guess, may you reply with 'No Snapshot is available'. "
    # )
    # text += (
    #     "When answering, do NOT just describe the image. Instead, write your answer as if you are telling someone the real answer to the question, in a complete sentence. "
    #     "For example, instead of 'Snapshot 0 A bowl is visible', you should write 'Snapshot 0 The fruit bowl is on the kitchen counter.' "
    # )
    # text += (
    #     "If, and only if, none of the Snapshots is sufficient, you can return: 'No Snapshot is available.' "
    # )
    # text += (
    #     "Note that if you choose a Snapshot to answer the question: "
    #     "(1) You must provide a clear and direct answer to the question that can be understood without referring to the image. "
    #     "Do not mention words like 'snapshot', 'on the left of the image', etc. "
    #     "You must only choose from the provided Snapshot indices. Do not make up an index that is not listed above. "
    # )
    # text += (
    #     "(2) You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot. "
    #     "Again, only choose from the provided Snapshot indices and do not create any indices that are not listed above. "
    # )

    # 2
    # text = "Please provide your answer in the following format: 'Snapshot i [Answer]' or 'No Snapshot is available', where i is the index of the snapshot you choose. "
    # text += "You should select one of the provided Snapshots and give a clear and direct answer to the question. Only reply 'No Snapshot is available' if it is truly impossible to answer from any Snapshot. "
    # text += "Write your answer as a complete sentence that directly responds to the question, not just a description of the image. Use simple and direct sentences, avoid vague or descriptive language. Do not mention words like 'snapshot', 'on the left of the image', etc. "
    # text += "For example, if you choose the first snapshot, you can return 'Snapshot 0 The fruit bowl is on the kitchen counter.'. "
    # text += "or if you choose the second snapshot, you can return 'Snapshot 1 Next to the fireplace'. "
    # text += "You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot."
    # text += "Note: Do not mention words like 'snapshot', 'in the image', or image positions. Only use the provided Snapshot indices, and do not make up any index that is not listed above. Only output the complete answer as a direct response, without any extra words, explanation, or reasoning."

    text += "Please answer in exactly one of the following two formats:\n"
    text += "1. Snapshot i [Your complete answer as a full sentence.]\n"
    text += "2. No Snapshot is available.\n"
    text += "The two formats are mutually exclusive. Never combine 'No Snapshot is available' with any Snapshot index.\n"
    text += "If you select a Snapshot, you must provide a clear and direct answer in a complete sentence.\n"
    text += "Only output your answer in one of the two formats above, with no extra words, explanation, or reasoning.\n"
    text += "Examples:\n"
    text += "Snapshot 0 The fruit bowl is on the kitchen counter.\n"
    text += "Snapshot 1 Next to the fireplace.\n"
    text += "No Snapshot is available.\n"
    text += "You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot."
    text += "Note: Do not mention words like 'snapshot', 'in the image', or image positions. Only use the provided Snapshot indices, and do not make up any index that is not listed above. Only output the complete answer as a direct response, without any extra words, explanation, or reasoning."


    content.append((text,))

    return sys_prompt, content




from collections import Counter
import random
import re
import logging

def call_openai_api_vote(sys_prompt, content, num_trials=5, max_tiebreak_rounds=5, seed=None):
    """
    Only for 'frontier' voting. Returns the most voted 'frontier <idx> ...' response.
    Minimal logging: only frontier index count and final chosen index.
    """
    tiebreak_round = 0
    candidate_indices = None
    while True:
        responses = []
        raw_indices = []
        for _ in range(num_trials):
            resp = call_openai_api(sys_prompt, content, seed=seed)
            if resp is not None:
                resp = resp.strip()
                m = re.match(r"frontier\s+(\d+)", resp.lower())
                if m:
                    idx = int(m.group(1))
                    if candidate_indices is None or idx in candidate_indices:
                        responses.append(resp)
                        raw_indices.append(idx)
        if not responses:
            logging.warning("[Frontier Voting] All responses are None. Return None.")
            return None
        # 只看 index 计数
        index_counter = Counter(raw_indices)
        log_str = " | ".join([f"frontier {idx}: {count}" for idx, count in index_counter.items()])
        logging.info(f"[Frontier Voting][Round {tiebreak_round+1}] {log_str}")
        max_count = max(index_counter.values())
        winners = [idx for idx, count in index_counter.items() if count == max_count]
        if len(winners) == 1:
            logging.info(f"[Frontier Voting] Selected: frontier {winners[0]}")
            # 找到第一个对应index的完整响应返回
            for resp in responses:
                m = re.match(r"frontier\s+(\d+)", resp.lower())
                if m and int(m.group(1)) == winners[0]:
                    return resp
        else:
            candidate_indices = winners
            tiebreak_round += 1
            if tiebreak_round >= max_tiebreak_rounds:
                chosen = random.choice(winners)
                logging.info(f"[Frontier Voting] Max tie-break rounds reached. Randomly selected: frontier {chosen}")
                for resp in responses:
                    m = re.match(r"frontier\s+(\d+)", resp.lower())
                    if m and int(m.group(1)) == chosen:
                        return resp






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


def get_prefiltering_classes(question, seen_classes, top_k=10, image_goal=None, seed=None):
    prefiltering_sys, prefiltering_content = format_prefiltering_prompt(
        question, sorted(list(seen_classes)), top_k=top_k, image_goal=image_goal
    )

    message = ""
    for c in prefiltering_content:
        message += c[0]
        if len(c) == 2:
            message += f": image {c[1][:10]}..."
    response = call_openai_api(prefiltering_sys, prefiltering_content, seed=seed)
    if response is None:
        return []

    # parse the response and return the top_k objects
    selected_classes = response.strip().split(" ")
    selected_classes = [cls.strip() for cls in selected_classes]
    selected_classes = [cls for cls in selected_classes if cls in seen_classes]
    selected_classes = selected_classes[:top_k]

    return selected_classes


def prefiltering(
    question, snapshot_classes, seen_classes, top_k=10, image_goal=None, verbose=False, seed=None
):
    selected_classes = get_prefiltering_classes(
        question, seen_classes, top_k, image_goal, seed=seed
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



def glm_answer(text):
    """
    提取<answer>标签后的内容。如果有闭合</answer>标签，提取两者之间的内容；
    如果没有闭合标签，则提取<answer>之后到行尾或字符串末尾的内容。
    不区分大小写。
    """
    # 先尝试标准闭合标签
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 没有闭合标签，则找<answer>到结尾
    match = re.search(r"<answer>\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # 都没有返回空
    return ""




def parse_frontier_index(output: str):
    """
    解析输出文本，返回(reason, index)
    支持全文任意位置的frontier index格式
    """
    matches = list(re.finditer(r'frontier\s*(\d+)', output, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        index = int(last_match.group(1))
        reason = output[:last_match.start()].strip()
        return reason, index
    else:
        raise ValueError(f"Could not parse frontier index")



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



def explore_step(step, cfg, verbose=False, chosen_frontier_path=None, step_idx=None):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
    
    # Get base_mode parameter from config
    base_mode = cfg.get('base_mode', 'hierarchy')  # default to 'hierarchy'
    chat_seed = cfg.get('chat_seed', None)  # Get chat_seed from config
    step["chat_seed"] = chat_seed  # Pass to get_step_info for prefiltering
    
    logging.info(f"[explore_step] base_mode={base_mode}, kmeans={cfg.get('kmeans', 'not set')}, chat_seed={chat_seed}")
    
    (
        question,
        image_goal,
        egocentric_imgs,
        frontier_imgs,
        frontier_imgs_0,
        frontier_imgs_1,
        snapshot_imgs,
        snapshot_classes,
        snapshot_id_mapping,
    ) = get_step_info(step, verbose)
    
    logging.info(f"[explore_step] frontier_imgs count: {len(frontier_imgs)}, frontier_imgs_0: {len(frontier_imgs_0)}, frontier_imgs_1: {len(frontier_imgs_1)}")

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

    if len(snapshot_imgs) == 0:
        print("No snapshot images available, directly entering frontier exploration.")
        
    else:
        print(f"Snapshot images available: {len(snapshot_imgs)}")
        retry_bound = 3
        for _ in range(retry_bound):
            full_response = call_openai_api(sys_prompt, content, seed=chat_seed)
            # full_response = glm_answer(full_response)  # 处理glm的输出格式
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
                    reason = clean_reason(" ".join(tokens[2:]).strip()) 
                    if 0 <= idx < len(snapshot_imgs) and reason != "":
                        response = f"{tokens[0]} {tokens[1]}"
                        # reason = " ".join(tokens[2:]).strip()
                        # reason = clean_reason(reason)  
                        return response, snapshot_id_mapping, reason, len(snapshot_imgs)
                    elif 0 <= idx < len(snapshot_imgs) and reason == "":
                        print(f"Snapshot index {tokens[1]} has no reason.")
                        continue
                    else:
                        print(f"Snapshot index out of range: {tokens[1]}")
                        continue
            elif "no snapshot is available" in full_response:
                # 明确拒绝，直接进入frontier
                break
            else:
                print(f"Unrecognized snapshot response: {full_response}")
                continue


    # ==== Step 2: frontier prompt (listwise or two-stage) ====
    retry_bound = 3

    context = None
    
    # Check if episodic context is enabled
    use_episodic_context = cfg.get('episodic_context', False) or "froncon" in cfg.exp_name
    
    if use_episodic_context:
        if not os.path.exists(chosen_frontier_path):
            os.makedirs(chosen_frontier_path, exist_ok=True)

        png_files = [f for f in os.listdir(chosen_frontier_path) if f.endswith('.png')]
        if len(png_files) > 0:
            sys_prompt, content = frontier_context(chosen_frontier_path)
            context = call_openai_api(sys_prompt, content, seed=chat_seed)
            logging.info(f"[Episodic Context] Generated context summary: {context}")
        else:
            logging.info(f"[Episodic Context] No previous frontiers yet, skipping context generation")

    # Check if using listwise mode
    if base_mode == "listwise":
        # Listwise mode: directly use all frontiers without hierarchy
        logging.info("[Listwise Mode] Using all frontiers without hierarchy")
        
        sys_prompt, content = format_explore_prompt_frontier(
            question,
            egocentric_imgs,
            frontier_imgs,   # All frontiers (layer0 + layer1)
            snapshot_imgs,
            snapshot_classes, 
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            context=context,
        )
        
        if verbose:
            logging.info(f"Input prompt (frontier listwise):")
            message = sys_prompt
            for c in content:
                message += c[0]
                if len(c) == 2:
                    message += f"[{c[1][:10]}...]"
            logging.info(message)
        
        idx_final = None
        for _ in range(retry_bound):
            full_response = call_openai_api(sys_prompt, content, seed=chat_seed)
            if full_response is None:
                print("call_openai_api (frontier listwise) returns None, retrying")
                continue
            if isinstance(full_response, list):
                full_response = " ".join(full_response)
            full_response = full_response.strip().lower()
            try:
                reason, idx_final = parse_frontier_index(full_response)
                if 0 <= idx_final < len(frontier_imgs):
                    response = f"frontier {idx_final}"
                    if chosen_frontier_path and os.path.exists(chosen_frontier_path):
                        save_base64_to_png(frontier_imgs[idx_final], chosen_frontier_path, step_idx, idx_final)
                    return response, snapshot_id_mapping, reason, len(snapshot_imgs)
                else:
                    print(f"Listwise frontier index out of range: {idx_final}")
            except Exception as e:
                print(f"Listwise frontier format error: {full_response} | {e}")
        
        # If parsing failed, random selection
        if idx_final is None or idx_final >= len(frontier_imgs):
            idx_random = random.randint(0, len(frontier_imgs) - 1)
            response = f'frontier {idx_random}'
            reason = f"Randomly selected index {idx_random} due to parsing failure."
            return response, snapshot_id_mapping, reason, len(snapshot_imgs)
    
    else:
        # Hierarchy mode: two-stage frontier selection
        # ------- Step 2.1: 先让VLM在layer0大簇里选 -------
        sys_prompt, content = format_explore_prompt_frontier(
            question,
            egocentric_imgs,
            frontier_imgs_0,   # layer0候选
            snapshot_imgs,
            snapshot_classes, 
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            context=context,
        )
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
            # full_response = call_openai_api_vote(sys_prompt, content)
            full_response = call_openai_api(sys_prompt, content, seed=chat_seed)
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
                    print(f"Layer0 index out of range: {idx0}")
            except Exception as e:
                print(f"Layer0 format error: {full_response} | {e}")
        # 如果idx0仍然是None，说明解析失败

        if idx0 is None:
            # return None, snapshot_id_mapping, None, len(snapshot_imgs)
            idx_random = random.randint(0, len(frontier_imgs_0) - 1)
            response = f'frontier {idx_random}'
            reason = f"Randomly selected index {idx_random} due to parsing failure."
            return response, snapshot_id_mapping, reason, len(snapshot_imgs)
        

        logging.info(f"[Layer0] VLM selected index: {idx0}")
        logging.info(f"reason for layer0 selection: {reason}")
        for k, v in step['layer0_to_layer1'].items():
            logging.info(f"  Layer0 {k}: {v}")
        # ------- Step 2.2: 在选中的layer0大簇下所有layer1细簇中选 -------
        full_response_layer0 = full_response.strip().lower()
        if idx0 not in step['layer0_to_layer1']:
            response = f"frontier {idx0}"
            final_reason = full_response.lower()
            logging.info(f"[Layer0] Layer0 index {idx0} has no corresponding layer1 subclusters. Directly returning layer0 as the frontier (global index: {idx0})")
            return response, snapshot_id_mapping, final_reason, len(snapshot_imgs)
        else:
            layer1_indices = step['layer0_to_layer1'][idx0]   # 例如 [1, 2]
            frontier_imgs_subgroup = [frontier_imgs_1[i] for i in layer1_indices]
            if len(layer1_indices) == 1:
                final_layer1_idx = layer1_indices[0]
                global_frontier_idx = len(step["frontier_imgs_0"]) + final_layer1_idx
                response = f"frontier {global_frontier_idx}"
                final_reason = "Only one candidate in this subcluster, selected by default."
                logging.info(f"[Layer1] Only one candidate ({global_frontier_idx}), selected by default.")
                save_base64_to_png(frontier_imgs_1[int(final_layer1_idx)], chosen_frontier_path, step_idx, final_layer1_idx)
                return response, snapshot_id_mapping, final_reason, len(snapshot_imgs)
            sys_prompt, content = format_explore_prompt_frontier(
                question,
                egocentric_imgs,
                frontier_imgs_subgroup,    # 只给当前大簇下的所有layer1细簇
                snapshot_imgs,
                snapshot_classes,
                egocentric_view=step.get("use_egocentric_views", False),
                use_snapshot_class=True,
                image_goal=image_goal,
                context=context,
            )
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
            

            idx1_in_subgroup = None
            final_reason = ""
            for _ in range(retry_bound):
                full_response = call_openai_api(sys_prompt, content, seed=chat_seed)
                # full_response = call_openai_api_vote(sys_prompt, content)
                
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
                        print(f"Layer1 index out of range: {idx1_in_subgroup}")
                except Exception as e:
                    print(f"Layer1 format error: {full_response} | {e}")
                    

            if idx1_in_subgroup is None:
                idx_random = random.randint(0, len(frontier_imgs_subgroup) - 1)
                response = f'frontier {idx_random}'
                reason = f"Randomly selected index {idx_random} due to parsing failure."
                return response, snapshot_id_mapping, reason, len(snapshot_imgs)
            
            elif idx1_in_subgroup >= len(layer1_indices):
                logging.warning(f"[Fallback] Invalid or missing Layer1 index ({idx1_in_subgroup}), fallback to Layer0 index {idx0}")
                response = f"frontier {idx0}"
                final_reason = full_response_layer0
                return response, snapshot_id_mapping, full_response_layer0, len(snapshot_imgs)

            final_layer1_idx = layer1_indices[idx1_in_subgroup]
            # frontier index = len(self.frontiers_layer0) + final_layer1_idx
            global_frontier_idx = len(step["frontier_imgs_0"]) + final_layer1_idx
            response = f"frontier {global_frontier_idx}"
            logging.info(f"[Layer1] VLM selected group index: {idx1_in_subgroup}")
            logging.info(f"[Layer1] This corresponds to global layer1 index: {final_layer1_idx} (global index: {global_frontier_idx})")

            save_base64_to_png(frontier_imgs_1[int(final_layer1_idx)], chosen_frontier_path, step_idx, final_layer1_idx)

            return response, snapshot_id_mapping, reason, len(snapshot_imgs)

