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
import json
import random
from src.context_generator import FrontierSimilaritySearcher
from src.context_generator import _build_searcher_if_ready, _resolve_episode_id, _process_candidate_one, run_layer0_recall_and_aggregate, run_layer1_recall_and_aggregate_for_subgroup
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
                model="qwen",  # gpt-4o-internvl-minicpm-qwen
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
            

    # 4 here is the frontier images
    text = "The followings are all the Frontiers that you can explore:  "
    content.append((text,))
    if len(frontier_imgs) == 0:
        content.append(("No Frontier is available ",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i} ", frontier_imgs[i]))
            

    # 5 here is the format of the answer
    text = "Please provide your answer in the following format: 'Snapshot i [Answer]' or 'Frontier i [Reason]', where i is the index of the snapshot or frontier you choose. "
    text += "For example, if you choose the first snapshot, you can return 'Snapshot 0 The fruit bowl is on the kitchen counter.'. "
    text += "If you choose the second frontier, you can return 'Frontier 1 I see a door that may lead to the living room.'. "
    text += "Note that if you choose a snapshot to answer the question, (1) you should give a direct answer that can be understood by others. Don't mention words like 'snapshot', 'on the left of the image', etc; You must only choose from the provided Snapshot or Frontier indices. Do not make up an index that is not listed above."
    text += "(2) you can also utilize other snapshots, frontiers and egocentric views to gather more information, but you should always choose one most relevant snapshot to answer the question. You must only choose from the provided Snapshot or Frontier indices. Do not make up an index that is not listed above. "
    content.append((text,))

    return sys_prompt, content


# v0
# def format_explore_prompt_frontier(
#     question,
#     egocentric_imgs,
#     frontier_imgs,
#     snapshot_imgs,
#     snapshot_classes,
#     egocentric_view=False,
#     use_snapshot_class=True,
#     image_goal=None,
#     context=None,          # <- keep as-is: this is your env_con (replay/recall)
#     episodic_con=None      # <- new: episodic context (per-episode frontier summary)
#     ):
#     # ========= System: Inputs/Outputs contract + recall & episodic consumption =========
#     sys_prompt = (
#         "Role: You are an agent that explores indoor scenes to answer a question by choosing exactly one Frontier.\n"
#         "You WILL BE GIVEN (as user content, in this order when available):\n"
#         "1) An optional recall context (a short natural-language recap of earlier exploration in similar scenes; it may include a TRANSFER summary of reusable visual cues).\n"
#         "2) An optional EPISODIC context (a factual summary of what has been explored so far in THIS episode, and what likely remains unexplored).\n"
#         "3) The current question (and possibly a goal image).\n"
#         "4) The current egocentric view (optional).\n"
#         "5) The list of Frontier candidate images, each with an index like 'Frontier 0', 'Frontier 1', ...\n"
#         "Your TASK: choose the single Frontier most helpful to make progress toward answering the current question.\n"
#         "Strict rules:\n"
#         "- Use ONLY the provided candidates; never invent images or indices.\n"
#         "- Refer to candidates ONLY by their indices (e.g., 'frontier 0'). Do NOT use titles, captions, or any labels from any context.\n"
#         "- If a recall context is provided, treat it as background cross-episode experience; extract transferable cues but prioritize the current question and visible evidence.\n"
#         "- If an EPISODIC context is provided, treat it as the current episode's factual state: use it to avoid redundant choices and to reason about explored vs. likely-unexplored directions; it is evidence, not an instruction.\n"
#         "- Be thorough in your reasoning: make your analysis explicit and structured before the final choice.\n"
#         "Your OUTPUT MUST include the following sections in order:\n"
#         "Step 0: List exactly the candidate indices you received (format 'Candidate indices: frontier 0, frontier 1, ...').\n"
#         "Step 1: For EACH candidate, describe what you see (objects, layout, cues relevant to the question). Only discuss candidates listed in Step 0.\n"
#         "Step 2: Compare candidates strictly from Step 0 for their relevance to the question, USING the contexts when available.\n"
#         "Step 3: Select the single most relevant candidate and justify your choice concisely.\n"
#         "FINAL: On a NEW line, output ONLY 'frontier i' (the chosen index) with nothing else.\n"
#     )

#     content = []

#     # ===== Recall context as user content (only if present; keep original logic) =====
#     has_context = bool(context and isinstance(context, str) and context.strip())
#     if has_context:
#         content.append(("Recall context (ENVIRONMENT / replay):\n" + context.strip(),))

#     # ===== EPISODIC context as user content (new, optional) =====
#     has_episodic = bool(episodic_con and isinstance(episodic_con, str) and episodic_con.strip())
#     if has_episodic:
#         content.append(("EPISODIC context (episode so far):\n" + episodic_con.strip(),))

#     # ===== Question (with optional goal image) =====
#     q_text = f"Question: {question}"
#     if image_goal is not None:
#         content.append((q_text, image_goal))
#     else:
#         content.append((q_text + " ",))

#     content.append(("Select the Frontier that would help find the answer of the question. ",))

#     # ===== Egocentric (guarded) =====
#     if egocentric_view and egocentric_imgs and len(egocentric_imgs) > 0:
#         content.append(("The following is the egocentric view of the agent in forward direction: ", egocentric_imgs[-1]))

#     # ===== Frontier candidates =====
#     content.append(("The following are all the Frontiers that you can explore:  ",))
#     if len(frontier_imgs) == 0:
#         content.append(("No Frontier is available",))
#     else:
#         for i in range(len(frontier_imgs)):
#             content.append((f"Frontier {i} ", frontier_imgs[i]))

#     # ===== CoT skeleton (kept compatible with your parser) =====
#     text = ""
#     text += "You are required to reason step by step and only output your final choice at the end. Please follow the instructions below carefully. "

#     # Step 0
#     text += "Step 0: List all candidate images you are given and their indices in the following format: 'Candidate indices: frontier 0, frontier 1, ...' (listing only the actual indices provided below; do NOT add, omit, or change any index). "
#     text += "You must ONLY discuss and compare the images whose indices are listed in Step 0. You are STRICTLY FORBIDDEN to invent, mention, analyze, or refer to any images or indices that are not explicitly listed in Step 0. "

#     # Step 1 — richer per-candidate observation
#     text += "Step 1: For each provided Frontier image, describe in detail what you see. Focus on visible objects, scene layout, and any clues relevant to the question. "
#     text += "Provide 2–3 sentences per candidate, and ONLY describe the images with the indices listed in Step 0. Start your answer with 'Step 1:' and describe each candidate separately. "
#     if has_context:
#         text += "When relevant, naturally note resemblance or contrast with the recall (ENVIRONMENT) using visual features only (do not use any titles or indices from the recall). "
#     if has_episodic:
#         text += "When relevant, refer to the EPISODIC context to avoid redundant exploration or to highlight likely-unexplored directions (do not invent any indices). "

#     # Step 2 — deeper, explicit dual-context use with labeled subparagraphs
#     text += "Step 2: Analyze what the question is asking for. Then, compare ONLY the frontiers listed in Step 0, by analyzing the clues shown in each image and their relevance to the question. Do NOT mention, analyze, or imagine any other indices. Start this section with 'Step 2:'. "
#     if has_context:
#         text += "Include a labeled subparagraph starting with 'Context reflection — ENVIRONMENT:' (3–4 sentences) where you extract 1–2 transferable visual cues from the recall (prefer cues named in its TRANSFER summary if present) and apply them explicitly to the current candidates by naming which 'frontier i' match or conflict with those cues and why, citing concrete visible features. "
#     if has_episodic:
#         text += "Include a labeled subparagraph starting with 'Context reflection — EPISODIC:' (3–4 sentences) where you state which directions appear already explored vs. likely unexplored, indicate potential redundancy, and explain how this affects your preferences among the Step‑0 candidates. "
#     if has_context or has_episodic:
#         text += "Then write a labeled 'Synthesis:' subparagraph (2–3 sentences) that reconciles any tension between ENVIRONMENT cues and EPISODIC constraints, and identifies the one or two leading candidates by naming the decisive visual features. "
#         text += "If the two contexts conflict, explicitly explain which one you prioritize and why (e.g., strong direct visual evidence may override a weak transferable cue). "
#     else:
#         text += "Provide a thorough comparison solely from current visual evidence (3–5 sentences). "
#     text += "Avoid generic statements; name specific features (e.g., doorway/threshold/outdoor light for entrances; readable faces for text/symbols; sink–cabinet–countertop grouping for kitchen). "

#     # Step 3 — justified choice + brief runner-up contrast
#     text += "Step 3: Based on your analysis above, select the single most relevant frontier for making progress toward answering the question. Clearly state your reasoning and why you select this one, but ONLY from the indices listed in Step 0. Begin this section with 'Step 3:'. "
#     if has_context or has_episodic:
#         text += "Tie your justification back to the extracted ENVIRONMENT cue(s) and/or the EPISODIC constraints; if you deviate from a cue, name it and justify the deviation using current evidence. "
#     text += "Briefly contrast your choice with the strongest runner‑up (1–2 sentences) to show why your chosen frontier better satisfies the question right now. "

#     # Final constraints
#     text += "After completing Step 3, output your final answer on a new line in the format: 'frontier i' (where i is one of the indices listed in Step 0). Do not include any other words, indices, or explanations on that line. "
#     text += "You MUST select one and only one of the provided Frontier indices listed in Step 0. You are NOT allowed to say that none is suitable or refuse to choose. "
#     text += "Choose the frontier that is MOST likely to help you answer the question, based ONLY on the visible clues, transferable cues, and episode-so-far constraints. "
#     text += "If you choose a frontier to answer the question: you should provide a clear and specific reason directly related to the question. "
#     text += "Do NOT mention words like 'frontier', directions, or image positions in your reasoning except when referring to the candidate indices listed in Step 0. Only use the provided Frontier indices; do NOT make up or analyze any index that is not listed above. "
#     text += "Only use the indices listed in Step 0. Any mention, analysis, or invention of other indices will be considered an error. Do NOT refer to images/frontiers not listed above."

#     content.append((text,))

#     return sys_prompt, content


# v1
def format_explore_prompt_frontier(
    question,
    egocentric_imgs,
    frontier_imgs,
    snapshot_imgs,
    snapshot_classes,
    egocentric_view=False,
    use_snapshot_class=True,
    image_goal=None,
    context=None,       # Experience replay text (cross-episode, similar-scene summaries)
    episodic_con=None   # Episodic context text (this episode: recent steps/path & seen/unseen summary)
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
    # System role & definitions
    # =========================
    sys_prompt = (
        "You are an embodied exploration agent. Your task is to pick exactly one frontier image as the next exploration direction, "
        "so that you can best progress toward answering the current question.\n"
        "FRONTIERS: Candidate entry points toward yet-unseen or information-rich regions—typical visual patterns include doorways/thresholds, corridors/intersections, "
        "stairs, corners/turns, or vantage points that likely open new coverage. Each candidate is an image referred to strictly by an index like 'frontier 0'. "
        "Use ONLY these indices. Never invent, omit, or rename indices, and never analyze images that are not provided.\n"
        "EGOCENTRIC VIEW (if shown): The agent’s immediate forward-looking camera view; use it as local context only.\n"
        "EPISODIC CONTEXT (if present): A factual textual summary of the recent steps within THIS episode—the path taken, what has been observed, "
        "and what likely remains unseen. Use this to avoid redundant choices and prefer novel, informative directions. It is evidence, not a command.\n"
        "EXPERIENCE REPLAY (if present): A textual summary retrieved from OTHER episodes in similar scenes. "
        "It describes what was observed there, which frontier was chosen, what actions followed, what outcome/reward resulted, and a brief critique of why that choice helped (or not). "
        "Extract only transferable visual patterns/strategies (e.g., typical object groupings, spatial layouts). If any replay hint conflicts with current visible evidence, "
        "always prioritize the current evidence.\n"
        "Your reasoning must be concrete and visual. Name specific objects, layouts, textures, lighting, text-bearing surfaces/symbols, "
        "and any cues directly relevant to the question. Reason first and answer last. On the final line, print ONLY the chosen index as 'frontier i'. "
        "You must select one of the provided candidates; do not say that none is suitable.\n"
    )

    content = []

    # =========================
    # Frontier candidates
    # =========================
    content.append(("Frontier candidates (the ONLY options you may choose):",))
    if len(frontier_imgs) == 0:
        content.append(("No frontier is available.",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"Frontier {i}", frontier_imgs[i]))

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
            "Experience replay — knowledge from OTHER episodes in similar scenes. "
            "It states what was observed, which frontier was chosen, what actions followed, what outcome/reward resulted, "
            "and a short critique explaining why that choice helped or hindered the final result. "
            "Extract only transferable visual patterns/strategies and prefer the current visible evidence when conflicts arise:\n"
            + context.strip(),
        ))

    # =========================
    # Question
    # =========================
    q_text = f"Question: {question}"
    if image_goal is not None:
        content.append((q_text, image_goal))
    else:
        content.append((q_text,))

    # =========================
    # CoT skeleton
    # =========================

    # Step 0
    text += "Step 0: List all candidate images you are given and their indices in the following format: 'Candidate indices: frontier 0, frontier 1, ...' (listing only the actual indices provided below; do NOT add, omit, or change any index). "
    text += "You must ONLY discuss and compare the images whose indices are listed in Step 0. You are STRICTLY FORBIDDEN to invent, mention, analyze, or refer to any images or indices that are not explicitly listed in Step 0. "

    # Step 1
    text += "Step 1: For each provided Frontier image, describe in detail what you see. Focus on visible objects, scene layout, and any clues relevant to the question. "
    text += "Provide 2–3 sentences per candidate, and ONLY describe the images with the indices listed in Step 0. Start your answer with 'Step 1:' and describe each candidate separately. "
    if has_experience:
        text += "Explicitly compare each candidate with the EXPERIENCE replay, noting resemblance or contrast using visual features only (do not copy titles or indices from the replay). "
    if has_episodic:
        text += "Explicitly compare each candidate with the EPISODIC context, showing whether it avoids redundancy or opens likely-unexplored directions. "

    # Step 2
    text += "Step 2: Analyze what the question is asking for. Then, compare ONLY the frontiers listed in Step 0, by analyzing the clues shown in each image and their relevance to the question. Do NOT mention, analyze, or imagine any other indices. Start this section with 'Step 2:'. "
    if has_experience:
        text += "Include a labeled subparagraph 'Context reflection — EXPERIENCE:' (3–4 sentences) where you extract 1–2 transferable cues from the replay and apply them explicitly to the current candidates by naming which 'frontier i' match or conflict with those cues and why, citing concrete visible features. "
    if has_episodic:
        text += "Include a labeled subparagraph 'Context reflection — EPISODIC:' (3–4 sentences) where you state which directions appear already explored vs. likely unexplored, indicate potential redundancy, and explain how this affects your preferences among the Step-0 candidates. "
    if has_experience or has_episodic:
        text += "Then write a labeled 'Synthesis:' subparagraph (2–3 sentences) that reconciles any tension between EXPERIENCE cues and EPISODIC constraints, and identifies the one or two leading candidates by naming the decisive visual features. "
        text += "If the two contexts conflict, explicitly explain which one you prioritize and why (e.g., strong direct visual evidence may override a weak transferable cue). "
    else:
        text += "Provide a thorough comparison solely from current visual evidence (3–5 sentences). "
    text += "Avoid generic statements; name specific features (e.g., doorway/threshold/outdoor light for entrances; readable faces for text/symbols; sink–cabinet–countertop grouping for kitchen). "

    # Step 3
    text += "Step 3: Based on your analysis above, select the single most relevant frontier for making progress toward answering the question. Clearly state your reasoning and why you select this one, but ONLY from the indices listed in Step 0. Begin this section with 'Step 3:'. "
    if has_experience or has_episodic:
        text += "Tie your justification back to the EXPERIENCE cue(s) and/or the EPISODIC constraints; if you deviate from a cue, name it and justify the deviation using current evidence. "
    text += "Briefly contrast your choice with the strongest runner-up (1–2 sentences) to show why your chosen frontier better satisfies the question right now. "

    # Final constraints
    text += "After completing Step 3, output your final answer on a new line in the format: 'frontier i' (where i is one of the indices listed in Step 0). Do not include any other words, indices, or explanations on that line. "
    text += "You MUST select one and only one of the provided Frontier indices listed in Step 0. You are NOT allowed to say that none is suitable or refuse to choose. "
    text += "Choose the frontier that is MOST likely to help you answer the question, based ONLY on the visible clues, transferable cues, and episode-so-far constraints. "
    text += "If you choose a frontier to answer the question: you should provide a clear and specific reason directly related to the question. "
    text += "Do NOT mention words like 'frontier', directions, or image positions in your reasoning except when referring to the candidate indices listed in Step 0. Only use the provided Frontier indices; do NOT make up or analyze any index that is not listed above. "
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

    content = []
    # 1 first is the question
    text = f"Question: {question}"
    if image_goal is not None:
        content.append((text, image_goal))
        
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
        

    # 3 here is the snapshot images
    text = "The followings are all the snapshots that you can choose (followed with contained object classes) "
    text += "Please note that the contained classes may not be accurate (wrong classes/missing classes) due to the limitation of the object detection model."
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

    # ---- 枚举所有可用index
    if len(snapshot_imgs) > 0:
        indices_list = ", ".join([str(i) for i in range(len(snapshot_imgs))])
        # 组合所有可选格式
        example_str = "', '".join([f"Snapshot {i}" for i in range(len(snapshot_imgs))])
        indices_hint = f"The only available Snapshot indices are: {indices_list}.\n"
        indices_example = f"You can answer using only '{example_str}', but never use an index not in this list.\n"
    else:
        indices_hint = ""
        indices_example = ""

    # ---- 插入提示到回答说明
    text = ""
    text += indices_hint
    text += indices_example
    text += "Please answer in exactly one of the following two formats:\n"
    text += "1. Snapshot i [Your complete answer as a full sentence.]\n"
    text += "2. No Snapshot is available.\n"
    text += (
        "If you select a Snapshot, you MUST always provide a complete, direct answer to the question in a full sentence. "
        "Never leave the answer blank or incomplete. Answers like 'Snapshot 2' alone are not allowed and will be considered invalid. "
        "Warning: If you output only 'Snapshot i' without a complete answer, your answer will be rejected and not considered valid.\n"
    )
    text += "The two formats are mutually exclusive. Never combine 'No Snapshot is available' with any Snapshot index.\n"
    text += "Only output your answer in one of the two formats above, with no extra words, explanation, or reasoning.\n"
    text += "Examples:\n"
    text += "Snapshot 0 The fruit bowl is on the kitchen counter.\n"
    if len(snapshot_imgs) > 1:
        text += f"Snapshot 1 Next to the fireplace.\n"
    text += "No Snapshot is available.\n"
    text += "You may also use information from other Snapshots and egocentric views to help you answer, but you must always select the single most relevant Snapshot.\n"
    text += "Only use the provided Snapshot indices, and DO NOT make up any index that is not listed above. Only output the complete answer as a direct response, without any extra words, explanation, or reasoning."

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





def save_base64_to_png(b64_str, save_dir, step_idx, idx):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{step_idx}-frontier{idx}.png")
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(BytesIO(img_bytes))
    img.save(save_path)
    return save_path


def save_base64_to_png_layer1(b64_str, save_dir, step_idx, idx, idx0):
    os.makedirs(save_dir, exist_ok=True)
    idx = idx%3
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
            parts = name.split('-')
            step = int(parts[0])
            fidx = '-'.join(parts[1:])  # 保留为字符串
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
        

    # 4. 明确只输出context summary，不要建议
    text = ""
    text += "Please output ONLY a single paragraph context summary, similar to the example above. "
    text += "Do NOT make suggestions or give next-step decisions."
    content.append((text,))

    return sys_prompt, content









def explore_step(step, cfg, verbose=False, chosen_frontier_path=None, step_idx=None):
    step["use_prefiltering"] = cfg.prefiltering
    step["top_k_categories"] = cfg.top_k_categories
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

    # ==== Step 2: two-stage frontier prompt ====
    retry_bound = 3

    # ==== (NEW) Layer-0 回忆与聚合（只对初始方向层做） ====
    _replay_top = int(getattr(cfg, "replay_top", 1))
    if _replay_top > 0:
        run_layer0_recall_and_aggregate(
            step=step,
            cfg=cfg,
            frontier_imgs_0=frontier_imgs_0,
            chosen_frontier_path=chosen_frontier_path,
            strategy=("random" if getattr(cfg, "replay_mode", "sim") == "random" else "sim"),
            top_k=_replay_top,
        )
    else:
        # 明确禁用 env 回放上下文
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
        context=(layer0_con if _replay_top > 0 else None),
        episodic_con=episodic_con,
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
                print(f"Layer0 index out of range: {idx0}")
        except Exception as e:
            print(f"Layer0 format error: {full_response} | {e}")
            
    if idx0 is None:
        idx_random = random.choice(frontier_imgs_0)
        response = f'frontier {idx_random}'
        reason = "no valid index found, randomly selected one."
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




        # —— 对该方向下的“更近处视角”子集做聚合（indices 为全局 layer1 索引） ——
        layer1_texts_all = step.get("replay_context_text_per_frontier", {}).get("layer1", [])
        layer1_context_text = None
        if layer1_texts_all and isinstance(layer1_indices, list) and len(layer1_indices) > 0:
            layer1_context_text = aggregate_recall_contexts_for_layer(
                layer_alias="closer looks",
                contexts=layer1_texts_all,
                indices=layer1_indices
            )



        # ==== (NEW) 对该方向的更近处子集做回忆与聚合 ====
        if _replay_top > 0:
            layer1_context_text = run_layer1_recall_and_aggregate_for_subgroup(
                step=step,
                cfg=cfg,
                frontier_imgs_1=frontier_imgs_1,
                layer1_indices=layer1_indices,
                chosen_frontier_path=chosen_frontier_path,
                strategy=("random" if getattr(cfg, "replay_mode", "sim") == "random" else "sim"),
                top_k=_replay_top,
            )
        else:
            layer1_context_text = None
            logging.info("[ReplayCtx] replay_top=0; skip layer1 subgroup recall and env context injection.")


        sys_prompt, content = format_explore_prompt_frontier(
            question,
            egocentric_imgs,
            frontier_imgs_subgroup,    # 只给当前大簇下的所有layer1细簇
            snapshot_imgs,
            snapshot_classes,
            egocentric_view=step.get("use_egocentric_views", False),
            use_snapshot_class=True,
            image_goal=image_goal,
            context=(layer1_context_text if _replay_top > 0 else None),
            episodic_con=episodic_con,
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
                    print(f"Layer1 index out of range: {idx1_in_subgroup}")
            except Exception as e:
                print(f"Layer1 format error: {full_response} | {e}")

        if idx1_in_subgroup is None:
            idx_random = random.choice(frontier_imgs_subgroup)
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

        save_base64_to_png_layer1(frontier_imgs_1[int(final_layer1_idx)], chosen_frontier_path, step_idx, final_layer1_idx, idx0)

        return response, snapshot_id_mapping, reason, len(snapshot_imgs)

