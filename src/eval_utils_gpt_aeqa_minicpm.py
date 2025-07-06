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
                model="minicpm",  # gpt-4o
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
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. To answer the question, you are required to choose a Frontier to further explore. "
    sys_prompt += "Definitions: "
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

    # 5 here is the format of the answer



    # 7
    text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index you choose. "
    text += "You must select one of the provided Frontier indices. Choose the frontier most likely to lead to the answer, and briefly explain why it is helpful for answering the question. "
    text += "For example: 'Frontier 1 There is a door that may lead to the kitchen, where the answer might be found.' "
    text += "Only use the provided indices. Do not make up new indices."






    # 6
    # text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index of the frontier you choose. "
    # text += "You MUST select one of the provided Frontier indices. Do NOT say that none is suitable or refuse to choose. "
    # text += "Choose the frontier that is most likely to help you answer the question, based on where the target object or information is likely to be found. "
    # text += "Give a short and specific reason directly related to the question. Do not use vague phrases such as 'to explore more' or 'see what is there'. "
    # text += "For example: 'Frontier 1 There is a doorway that may lead to the kitchen, where the object in the question could be.' "
    # text += "Only use the provided Frontier indices. Do not invent any index that is not listed above."



    # 5
    # text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index of the frontier you choose. "
    # text += (
    #     "You MUST select one and only one of the provided Frontier indices. You are NOT allowed to say that none is suitable, or to refuse to choose. "
    # )
    # text += (
    #     "You should select the frontier that is MOST likely to lead you closer to answering the question, based on visible clues, semantic hints, or the likely location of the target object. "
    #     "Your reasoning should connect the current question to what you can see or infer from the frontier images, focusing on which direction seems most promising for finding the information needed to answer. "
    # )
    # text += (
    #     "For example, if you choose the second frontier, you can return: 'Frontier 1 There is a door that may lead to the kitchen, which is likely to have the answer.' "
    # )
    # text += (
    #     "Note that when you choose a frontier to answer the question: (1) You should provide a clear and specific reason related to the question. Do not mention words like 'frontier', 'on the left of the image', etc. "
    #     "You must only choose from the provided Frontier indices. Do not make up an index that is not listed above. "
    # )
    # text += (
    #     "(2) You may also consider information from other frontiers and egocentric views to help your decision, but you must always select the single most relevant frontier for progressing towards answering the question. Again, only choose from the provided Frontier indices and do not create any indices that are not listed above. "
    # )

    # 4
    # text = "Please provide your answer in the following format: 'Frontier i [Reason]', where i is the index of the frontier you choose. "
    # text += "You MUST select one and only one of the provided Frontier indices. You are NOT allowed to say that none is suitable or refuse to choose. "
    # text += "Choose the frontier that is MOST likely to help you answer the question, based on visible clues, semantic hints, or where the target object is likely to be found. "
    # text += "Your reasoning should clearly connect the question with what you observe or infer from the frontier images, focusing on which direction is most promising for finding the needed information. "
    # text += "For example, if you choose the second frontier, you can return: 'Frontier 1 There is a door that may lead to the kitchen, which is likely to have the answer.' "
    # text += "If you choose a frontier to answer the question: you should provide a clear and specific reason directly related to the question. Do not mention words like 'frontier', directions, or image positions. Only use the provided Frontier indices; do not make up an index that is not listed above. "
    # text += "You may also use information from other frontiers and egocentric views to help your decision, but always select the single most relevant frontier for making progress toward answering the question. Only choose from the provided Frontier indices and do not create any indices that are not listed above. "



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

    # 0 
    # text = "Please provide your answer in the following format: 'Snapshot i [Answer]' or 'No Snapshot is available', where i is the index of the snapshot you choose. "
    # text += "You should select one of the provided Snapshots and give a clear and direct answer to the question. Only reply 'No Snapshot is available' if it is truly impossible to answer from any Snapshot. "
    # text += "Write your answer as a complete sentence that directly responds to the question, not just a description of the image. Do not mention words like 'snapshot', 'on the left of the image', etc. "
    # text += "For example, if you choose the first snapshot, you can return 'Snapshot 0 The fruit bowl is on the kitchen counter.'. "
    # text += "You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot."
    # text += "Note: Do not mention words like 'snapshot', 'in the image', or image positions. Only use the provided Snapshot indices, and do not make up any index that is not listed above. "


    # 1
    text = "Please provide your answer in exactly one of the following two formats: 'Snapshot i [Your complete answer as a full sentence]' or 'No Snapshot is available'. "
    text += "If you select a Snapshot, you must provide a clear and direct answer to the question in a complete sentence, based on the chosen Snapshot. "
    text += "You may only reply 'No Snapshot is available' if it is truly impossible to answer the question from any Snapshot. "
    text += "Do not combine 'No Snapshot is available' with a Snapshot index. Never respond with 'Snapshot i No Snapshot is available'. "
    text += "Your answer must be a complete sentence that directly answers the question, not just a description of the image. "
    text += "For example, if you choose the first snapshot, your answer should be: 'Snapshot 0 The fruit bowl is on the kitchen counter.' "
    text += "Or if you find no Snapshot suitable, you can simply say 'No Snapshot is available'. "
    text += "You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot."
    text += "Note: Do not mention words like 'snapshot', 'in the image', or image positions. Only use the provided Snapshot indices, and do not make up any index that is not listed above. "



    content.append((text,))


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


# def explore_step(step, cfg, verbose=False):
#     step["use_prefiltering"] = cfg.prefiltering
#     step["top_k_categories"] = cfg.top_k_categories
#     (
#         question,
#         image_goal,
#         egocentric_imgs,
#         frontier_imgs,
#         snapshot_imgs,
#         snapshot_classes,
#         snapshot_id_mapping,
#     ) = get_step_info(step, verbose)
#     sys_prompt, content = format_explore_prompt(
#         question,
#         egocentric_imgs,
#         frontier_imgs,
#         snapshot_imgs,
#         snapshot_classes,
#         egocentric_view=step.get("use_egocentric_views", False),
#         use_snapshot_class=True,
#         image_goal=image_goal,
#     )

#     if verbose:
#         logging.info(f"Input prompt:")
#         message = sys_prompt
#         for c in content:
#             message += c[0]
#             if len(c) == 2:
#                 message += f"[{c[1][:10]}...]"
#         logging.info(message)

#     retry_bound = 3
#     final_response = None
#     final_reason = None
#     for _ in range(retry_bound):
#         full_response = call_openai_api(sys_prompt, content)

#         if full_response is None:
#             print("call_openai_api returns None, retrying")
#             continue


#         # 如果 full_response 是 token list（vLLM 的返回格式），先拼成字符串
#         if isinstance(full_response, list):
#             full_response = " ".join(full_response)

#         # 去掉前后空格
#         full_response = full_response.strip()

#         # 拆分 token 提取结果和理由
#         tokens = full_response.split()
#         if len(tokens) >= 2:
#             response = f"{tokens[0]} {tokens[1]}"
#             reason = " ".join(tokens[2:]).strip()
#         else:
#             print(f"Error in splitting response: {full_response}")
#             continue

#         response = response.lower()

#         try:
#             choice_type, choice_id = response.split(" ")
#         except Exception as e:
#             print(f"Error in splitting response: {response}")
#             print(e)
#             continue


#         response_valid = False
#         if (
#             choice_type == "snapshot"
#             and choice_id.isdigit()
#             and 0 <= int(choice_id) < len(snapshot_imgs)
#         ):
#             response_valid = True
#         elif (
#             choice_type == "frontier"
#             and choice_id.isdigit()
#             and 0 <= int(choice_id) < len(frontier_imgs)
#         ):
#             response_valid = True

#         if response_valid:
#             final_response = response
#             final_reason = reason
#             break

#     return final_response, snapshot_id_mapping, final_reason, len(snapshot_imgs)









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



def format_frontier_single_prompt(
    question,
    egocentric_imgs,
    frontier_img,
    egocentric_view=False,
    image_goal=None,
):
    sys_prompt = "Task: You are an agent in an indoor scene tasked with answering questions by observing the surroundings and exploring the environment. "
    sys_prompt += "To answer the question, you are required to judge whether this Frontier should be selected to further explore. "
    sys_prompt += "Definitions: "
    sys_prompt += "Frontier: An observation of an unexplored region that could potentially lead to new information for answering the question. "
    sys_prompt += "Selecting a frontier means that you will further explore that direction. "

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




def explore_step(step, cfg, verbose=False):
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
    for i, frontier_img in enumerate(frontier_imgs):
        sys_prompt, content = format_frontier_single_prompt(
            question,
            egocentric_imgs,
            frontier_img,
            egocentric_view=step.get("use_egocentric_views", False),
            image_goal=image_goal,
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
            # full_response = call_openai_api_vote(sys_prompt, content)
            full_response = call_openai_api(sys_prompt, content)
            if full_response is None:
                print("call_openai_api (single frontier) returns None, retrying")
                continue
            resp = full_response.strip().lower()
            lines = [line.strip() for line in resp.split('\n') if line.strip()]
            first_line = lines[0] if len(lines) > 0 else ""
            if first_line == "yes":
                reason = " ".join(lines[1:]) if len(lines) > 1 else ""
                reason = clean_reason(reason)
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
        full_response = call_openai_api(sys_prompt, content)
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
                    return response, snapshot_id_mapping, reason, len(snapshot_imgs)
                else:
                    print(f"Frontier index out of range: {tokens[1]}")
                    continue
        else:
            print(f"Unrecognized frontier response: {full_response}")
            continue

    # 如果兜底也失败，返回None
    return None, snapshot_id_mapping, None, len(snapshot_imgs)