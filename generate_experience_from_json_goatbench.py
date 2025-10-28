import os
import json
import base64
import argparse
import logging
from typing import Dict, Any, Optional, List
import re

import openai
from openai import OpenAI
from src.const import *

client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)


"""
This script intentionally avoids global config objects. Paths are passed explicitly.
"""


class FrontierSimilaritySearcher:
    def __init__(self, output_parent_dir: str, exp_name: str, questions_list_path: Optional[str], replay_json_path: str, method: str = "ahash", feature_fn=None):
        self.output_parent_dir = output_parent_dir
        self.exp_name = exp_name
        self.replay_json_path = replay_json_path

    # ---------- hashing utils ----------
def _best_match_l0_rel(target_rel: Optional[str], candidates: List[str]) -> Optional[str]:
    if not target_rel or not isinstance(target_rel, str) or not candidates:
        return None
    if target_rel in candidates:
        return target_rel
    try:
        # Expect formats like 'frontier/32-layer0-0.png'
        if "-layer0-" in target_rel:
            prefix, tail = target_rel.split("-layer0-", 1)
            # Build base prefix including '-layer0-'
            base = prefix + "-layer0-"
            # Try +/- 1 swap if numeric tail
            m = re.match(r"(\d+)\.png$", tail)
            if m:
                idx = int(m.group(1))
                for alt in (idx ^ 1, idx + 1, idx - 1):  # try toggle 0/1, then neighbors
                    probe = f"{base}{max(alt,0)}.png"
                    if probe in candidates:
                        return probe
            # Fallback: find any candidate sharing same base
            for c in candidates:
                if c.startswith(base):
                    return c
    except Exception:
        pass
    return None


def _best_match_l1_rel(target_rel: Optional[str], candidates: List[str]) -> Optional[str]:
    if not target_rel or not isinstance(target_rel, str) or not candidates:
        return candidates[0] if candidates else None
    if target_rel in candidates:
        return target_rel
    try:
        # Expect formats like 'frontier/32-layer1-0_1.png'
        if "-layer1-" in target_rel:
            prefix, tail = target_rel.split("-layer1-", 1)
            base = prefix + "-layer1-"
            m = re.match(r"(\d+)_([\d]+)\.png$", tail)
            major = None
            if m:
                major = int(m.group(1))
            # Prefer same major index if possible
            same_major = []
            for c in candidates:
                if c.startswith(base):
                    if major is None:
                        same_major.append(c)
                    else:
                        m2 = re.search(r"-layer1-(\d+)_", c)
                        if m2 and int(m2.group(1)) == major:
                            same_major.append(c)
            if same_major:
                return same_major[0]
            # Else fallback to first candidate
            return candidates[0]
    except Exception:
        pass
    return candidates[0] if candidates else None


def _parse_bvf_idx_from_rel(rel: Optional[str]) -> Optional[int]:
    try:
        if not rel or not isinstance(rel, str):
            return None
        m = re.search(r"/([0-9]+)-layer0-([0-9]+)\.png$", rel)  # TODO
        if m:
            return int(m.group(2))
    except Exception:
        pass
    return None


def _parse_cvf_idx_from_rel(rel: Optional[str]) -> Optional[int]:
    try:
        if not rel or not isinstance(rel, str):
            return None
        m = re.search(r"/([0-9]+)-layer1-([0-9]+)_([0-9]+)\.png$", rel) # TODO
        if m:
            return int(m.group(3))
    except Exception:
        pass
    return None


def format_content(contents):
    formatted_content = []
    for c in contents:
        formatted_content.append({"type": "text", "text": c[0]})
        if len(c) == 2:
            formatted_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{c[1]}",
                        "detail": "high",
                    },
                }
            )
    return formatted_content


def call_openai_api(sys_prompt, contents) -> Optional[str]:
    max_tries = 5
    retry_count = 0
    formatted_content = format_content(contents)
    message_text = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": formatted_content},
    ]
    while retry_count < max_tries:
        try:
            completion = client.chat.completions.create(
                model="qwen",
                messages=message_text,
                temperature=0.7,
                max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
            )
            return completion.choices[0].message.content
        except openai.RateLimitError:
            logging.info("Rate limit error, waiting before retry")
            import time
            time.sleep(30)
            retry_count += 1
            continue
        except Exception as e:
            logging.warning(f"OpenAI API error: {e}")
            import time
            time.sleep(60)
            retry_count += 1
            continue
    return None


def generate_captions_for_frontiers(
    searcher: "FrontierSimilaritySearcher",
    question_id: str,
    question_text: str,
    step_idx_text: str,
    initial_rels: List[str],
    detail_rels: List[str],
) -> Optional[str]:
    """
    Produce captions for BVFs/CVFs using the specified format and guidance.
    Expected output format (no extra text, exactly the following lines structure):
      BVF1: <description>
      BVF2: <description>
      ...
      CVF1: <description>
      CVF2: <description>
      ...
    """
    n_initial = len(initial_rels)
    n_detail = len(detail_rels)
    rules_parts: List[str] = []
    if n_initial > 0:
        rules_parts.append(
            f"You MUST output exactly {n_initial} BVF lines labeled BVF0..BVF{n_initial-1}. "
        )
    else:
        rules_parts.append("You MUST output no BVF lines. ")
    if n_detail > 0:
        rules_parts.append(
            f"You MUST output exactly {n_detail} CVF lines labeled CVF0..CVF{n_detail-1}. "
        )
    else:
        rules_parts.append("You MUST output no CVF lines. ")
    rules_str = "".join(rules_parts)
    sys_prompt = (
        "You are an embodied agent located in an indoor environment. You are given a question to answer and must perceive the environment with a camera and choose a frontier where to move in the next step to solve this task as efficiently as possible. "
        "At each step, you are firstly given a list of 'broad-view frontiers' (BVF), which are coarse exploration directions that cover distinct areas of the scene, and choose one index of BVF to look closer. "
        "With the selected BVF direction, you do not move; instead, you break down the view along this direction into closer, narrowed‑down views and are provided with a list of 'closer‑view frontiers' (CVF). CVF are detailed snapshots in the selected direction; the final movement target is the chosen CVF. "
        "Now, you are given the images of BVFs and CVFs at one step and the current question; your task is to describe each snapshot image and output EXACTLY in the following format (no extra text):\n\n"
        "BVF0: <description>\n\nBVF1: <description>\n\n…\n\nCVF0: <description>\n\nCVF1: <description>\n\n…\n\n"
        f"RULE: Each description is 4–6 sentences. Focus on visual elements relevant to solving the question and guiding exploration. {rules_str}"
        "Do not output any lists or headings beyond the specified lines; write only those BVF*/CVF* lines."
    )

    content = []
    # User content with question and labeled images
    content.append(("The current question you are solving is: ",))
    content.append((question_text or "",))
    if step_idx_text and step_idx_text != "NA":
        content.append((f"StepIndex: {step_idx_text}",))
    content.append(("You are given the following BVFs and CVFs:",))

    def _add_img(rel_path: str):
        abs_path = os.path.join(searcher.output_parent_dir, searcher.exp_name, question_id, rel_path)
        if os.path.exists(abs_path):
            with open(abs_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            content.append(("", b64))

    # Attach BVFs with labels BVF0:, BVF1:, ...
    for i, rel in enumerate(initial_rels, start=0):
        content.append((f"BVF{i}: ",))
        _add_img(rel)
    # Attach CVFs with labels CVF0:, CVF1:, ...
    for j, rel in enumerate(detail_rels, start=0):
        content.append((f"CVF{j}: ",))
        _add_img(rel)

    # Validate captions coverage (indices and counts) and retry up to 3 times if mismatch
    def _captions_match(text: Optional[str]) -> bool:
        try:
            if text is None:
                return False
            pat_bvf = re.compile(r"^\s*(?:bvf|bf)\s*([0-9]+)\s*[:：]", re.IGNORECASE)
            pat_cvf = re.compile(r"^\s*(?:cvf|cf)\s*([0-9]+)\s*[:：]", re.IGNORECASE)
            seen_bvf = set()
            seen_cvf = set()
            for raw in text.splitlines():
                line = raw.strip()
                if not line:
                    continue
                m = pat_bvf.match(line)
                if m:
                    try:
                        idx = int(m.group(1))
                        if 0 <= idx < n_initial:
                            seen_bvf.add(idx)
                        elif 1 <= idx <= n_initial:
                            seen_bvf.add(idx - 1)
                    except Exception:
                        pass
                    continue
                m = pat_cvf.match(line)
                if m:
                    try:
                        idx = int(m.group(1))
                        if 0 <= idx < n_detail:
                            seen_cvf.add(idx)
                        elif 1 <= idx <= n_detail:
                            seen_cvf.add(idx - 1)
                    except Exception:
                        pass
                    continue
            return (len(seen_bvf) == n_initial) and (len(seen_cvf) == n_detail)
        except Exception:
            return False

    last = None
    for attempt in range(3):
        last = call_openai_api(sys_prompt, content)
        if _captions_match(last):
            return last
        logging.warning(
            f"[CaptionValidate] mismatch on attempt {attempt+1}: step={step_idx_text}, q={question_id}, n_bvf={n_initial}, n_cvf={n_detail}"
        )
    # return the last attempt even if still mismatched
    return last



def _parse_experience_text(experience_text: Optional[str]) -> (str, str):
    """
    从 experience 字符串中解析 Critique 与 Abstraction。
    期望格式：
      "Critique: ...\nAbstraction: ..."
    返回 (critique, abstraction)，若解析失败则返回空串。
    """
    if not experience_text:
        return "", ""
    try:
        text = experience_text.strip()
        # 找到 "Critique:" 的起始
        crit_idx = text.lower().find("critique:")
        abs_idx = text.lower().find("abstraction:")
        critique = ""
        abstraction = ""
        if crit_idx != -1 and abs_idx != -1 and abs_idx > crit_idx:
            critique = text[crit_idx + len("Critique:"):abs_idx].strip()
            abstraction = text[abs_idx + len("Abstraction:"):].strip()
        elif abs_idx != -1:
            # 只有 Abstraction
            abstraction = text[abs_idx + len("Abstraction:"):].strip()
        elif crit_idx != -1:
            # 只有 Critique
            critique = text[crit_idx + len("Critique:"):].strip()
        return critique, abstraction
    except Exception:
        return "", ""


def generate_caption_tuples(
    input_json_path: str,
    output_parent_dir: str,
    exp_name: str,
    experience_json_path: str,
    tuple_output_json_path: str,
    questions_json_path: Optional[str] = "/home/hpc/v100dd/v100dd12/code/3D-Mem/data/aeqa_questions-168.json",
) -> Dict[str, Any]:
    """
    仅生成 Caption，并对齐 / 解析给定 experience_output.json 中的 question_id / step，
    将 {Critique, Abstraction} 从该 JSON 的 experience 文本中解析出来，与 Caption 一并写入：

    输出结构：
    {
      question_id: {
        "step_0": {
          "current_step": int,
          "total_step": int,
          "final_reward": "pass"|"fail",
          "Caption": str,
          "Critique": str,
          "Abstraction": str
        },
        ...
      },
      ...
    }
    """
    # 载入输入与参考 experience JSON
    with open(input_json_path, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    with open(experience_json_path, "r", encoding="utf-8") as f:
        exp_data = json.load(f)

    # 载入 questions 映射（question_id -> question）
    questions_map: Dict[str, str] = {}
    try:
        if questions_json_path and os.path.exists(questions_json_path):
            with open(questions_json_path, "r", encoding="utf-8") as f:
                qraw = json.load(f)
            if isinstance(qraw, list):
                for item in qraw:
                    if isinstance(item, dict):
                        qid = item.get("question_id")
                        qtext = item.get("question")
                        if isinstance(qid, str) and isinstance(qtext, str):
                            questions_map[qid] = qtext
    except Exception as e:
        logging.warning(f"[Questions] Failed to load questions from {questions_json_path}: {e}")

    # 构造搜索器以访问图像路径
    searcher = _ensure_searcher(output_parent_dir, exp_name, None, input_json_path)
    if searcher is None:
        raise FileNotFoundError(f"Cannot create searcher, input JSON not found: {input_json_path}")

    results: Dict[str, Any] = {}

    # 错误日志文件路径（与 tuple 输出同目录）并清空旧内容
    error_log_path = os.path.join(os.path.dirname(os.path.abspath(tuple_output_json_path)), "exp_error.txt")
    try:
        os.makedirs(os.path.dirname(error_log_path), exist_ok=True)
        with open(error_log_path, "w", encoding="utf-8") as ef:
            ef.write("")
    except Exception:
        pass

    def _append_error(msg: str) -> None:
        try:
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(msg.rstrip() + "\n")
        except Exception:
            pass

    # 遍历参考 exp_data，确保严格对齐 question_id 与 step
    for episode_id, qdict in (exp_data or {}).items():
        if not isinstance(qdict, dict):
            continue
        for question_id, qinfo_exp in qdict.items():
            if not isinstance(qinfo_exp, dict):
                continue

            steps_exp = (qinfo_exp.get("steps", {}) or {})
            if not isinstance(steps_exp, dict):
                continue

            # 在 input_data 中找到对应 question 的结构
            qinfo_in = (
                (input_data.get(episode_id, {}) or {}).get(question_id, {})
                if isinstance(input_data, dict) else {}
            )
            steps_in = qinfo_in.get("steps", {}) or {}
            question_text = (qinfo_in.get("question", "") or "").strip()
            final_reward = (qinfo_in.get("final_reward", None) or "").strip().lower()

            total_steps = len(steps_exp)

            # 初始化输出容器（以 question_id 为键），并写入顶层 question 文本
            results.setdefault(question_id, {})
            if "question" not in results[question_id]:
                q_text_from_map = questions_map.get(question_id)
                if not isinstance(q_text_from_map, str) or not q_text_from_map:
                    q_text_from_map = (qinfo_in.get("question", "") or "").strip()
                results[question_id]["question"] = q_text_from_map

            for step_key, step_exp in steps_exp.items():
                if not isinstance(step_exp, dict):
                    continue

                # 解析 current_step
                try:
                    current_step = int(str(step_key).split("_")[-1])
                except Exception:
                    current_step = -1

                # 从 input_data 中找到同名 step 的 frontier 信息
                step_in = steps_in.get(step_key, {}) if isinstance(steps_in, dict) else {}
                if isinstance(steps_in, dict) and step_key not in steps_in:
                    _append_error(f"[ALIGN_MISS] epi={episode_id} q={question_id} step={step_key} not found in input_json steps")
                frontier = step_in.get("frontier", {}) or {}
                if not isinstance(frontier, dict):
                    frontier = {}
                    _append_error(f"[FRONTIER_TYPE] epi={episode_id} q={question_id} step={step_key} frontier not dict; coerced to {{}}")

                chosen_frontier = step_in.get("chosen_frontier", {}) or {}
                chosen_l0 = chosen_frontier.get("layer0")
                chosen_l1 = chosen_frontier.get("layer1")

                initial_keys = list(frontier.keys())
                initial_rels = [os.path.join("frontier", k) for k in initial_keys]

                # Effective chosen rels after fallback correction
                chosen_l0_eff = chosen_l0
                chosen_l1_eff = chosen_l1

                # Determine details list by chosen_l0 (with fallback if needed)
                initial_key_for_details = None
                if chosen_l0_eff and isinstance(chosen_l0_eff, str) and chosen_l0_eff.startswith("frontier/"):
                    initial_key_for_details = chosen_l0_eff.split("/", 1)[1]
                detail_rels = frontier.get(initial_key_for_details, []) if initial_key_for_details else []

                # If no detail_rels but we do have a chosen_l1, try to locate its owning layer0 and switch
                if chosen_l1_eff and not detail_rels:
                    # Build l1 -> owning k map and a flat candidate list
                    l1_to_k = {}
                    all_l1 = []
                    for k in initial_keys:
                        for rel in (frontier.get(k, []) or []):
                            l1_to_k[rel] = k
                            all_l1.append(rel)
                    best_l1 = _best_match_l1_rel(chosen_l1_eff, all_l1)
                    if best_l1 and best_l1 in l1_to_k:
                        owning_k = l1_to_k[best_l1]
                        detail_rels = frontier.get(owning_k, []) or []
                        chosen_l0_eff = os.path.join("frontier", owning_k)
                # If chosen_l0 is not in initial_rels, try correcting it to the closest one
                if chosen_l0_eff and chosen_l0_eff not in initial_rels:
                    alt_l0 = _best_match_l0_rel(chosen_l0_eff, initial_rels)
                    if alt_l0:
                        chosen_l0_eff = alt_l0
                        initial_key_for_details = chosen_l0_eff.split("/", 1)[1]
                        detail_rels = frontier.get(initial_key_for_details, []) or detail_rels
                # If chosen_l1 not in details, try best match within current details
                if chosen_l1_eff and detail_rels and (chosen_l1_eff not in detail_rels):
                    alt_l1 = _best_match_l1_rel(chosen_l1_eff, detail_rels)
                    if alt_l1:
                        chosen_l1_eff = alt_l1
                if chosen_l1 and not detail_rels:
                    _append_error(f"[DETAIL_EMPTY] epi={episode_id} q={question_id} step={step_key} has chosen_l1 but no detail_rels under chosen_l0")

                # 选择索引：优先从文件名解析（严格 0 基），解析失败再回退列表索引
                def _idx0_l0(rel: Optional[str], rels: List[str]) -> int:
                    p = _parse_bvf_idx_from_rel(rel)
                    if p is not None:
                        return p
                    try:
                        return rels.index(rel) if rel and (rel in rels) else -1
                    except Exception:
                        return -1
                def _idx0_l1(rel: Optional[str], rels: List[str]) -> int:
                    p = _parse_cvf_idx_from_rel(rel)
                    if p is not None:
                        return p
                    try:
                        return rels.index(rel) if rel and (rel in rels) else -1
                    except Exception:
                        return -1

                selected_l0_index = _idx0_l0(chosen_l0 or chosen_l0_eff, initial_rels)
                selected_l1_index = _idx0_l1(chosen_l1 or chosen_l1_eff, detail_rels)

                # 不允许 -1：尝试基于文件名解析并夹取到有效范围
                def _sanitize_idx(idx: int, n: int, rel: Optional[str], parse_fn) -> int:
                    try:
                        p = parse_fn(rel)
                    except Exception:
                        p = None
                    if isinstance(p, int):
                        idx = p
                    if n <= 0:
                        return 0
                    if idx is None or idx < 0:
                        return 0
                    if idx >= n:
                        return n - 1
                    return idx

                selected_l0_index = _sanitize_idx(selected_l0_index, len(initial_rels), chosen_l0 or chosen_l0_eff, _parse_bvf_idx_from_rel)
                selected_l1_index = _sanitize_idx(selected_l1_index, len(detail_rels), chosen_l1 or chosen_l1_eff, _parse_cvf_idx_from_rel)
                if chosen_l0 and selected_l0_index < 0:
                    _append_error(f"[INDEX_MISS_BVF] epi={episode_id} q={question_id} step={step_key} chosen_l0={chosen_l0} not in initial_rels={initial_rels}")
                if chosen_l1 and selected_l1_index < 0:
                    _append_error(f"[INDEX_MISS_CVF] epi={episode_id} q={question_id} step={step_key} chosen_l1={chosen_l1} not in detail_rels={detail_rels}")

                # 生成 Caption（仅使用 Caption，不做 critique 生成）
                captions_text = None
                try:
                    step_idx_text = str(current_step) if current_step >= 0 else "NA"
                    captions_text = generate_captions_for_frontiers(
                        searcher=searcher,
                        question_id=question_id,
                        question_text=question_text,
                        step_idx_text=step_idx_text,
                        initial_rels=initial_rels,
                        detail_rels=detail_rels,
                    )
                except Exception as e:
                    logging.warning(f"[Caption] Failed: epi={episode_id} q={question_id} step={step_key} err={e}")
                    _append_error(f"[CAPTION_FAIL] epi={episode_id} q={question_id} step={step_key} err={e}")
                    captions_text = None
                if not captions_text:
                    _append_error(f"[CAPTION_EMPTY] epi={episode_id} q={question_id} step={step_key} captions is empty")

                # 从参考 exp 中解析 Critique / Abstraction
                experience_text = step_exp.get("experience") if isinstance(step_exp, dict) else None
                critique_text, abstraction_text = _parse_experience_text(experience_text if isinstance(experience_text, str) else None)
                if (not critique_text and not abstraction_text) and experience_text:
                    _append_error(f"[PARSE_EXPERIENCE_EMPTY] epi={episode_id} q={question_id} step={step_key} unable to parse critique/abstraction")

                # 写入输出结构（按 question_id 聚合）
                results[question_id][step_key] = {
                    "current_step": current_step if current_step >= 0 else 0,
                    "total_step": total_steps,
                    "final_reward": final_reward,
                    "Caption": (captions_text or "").strip(),
                    "Critique": critique_text,
                    "Abstraction": abstraction_text,
                    "chosen_BVF": selected_l0_index,
                    "chosen_CVF": selected_l1_index,
                }

                # 增量写盘，避免长任务中断丢失
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(tuple_output_json_path)), exist_ok=True)
                    with open(tuple_output_json_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    logging.info(f"[Save] Wrote progress to {tuple_output_json_path}")
                except Exception as e:
                    logging.warning(f"[Save] Failed to write progress: {e}")
                    _append_error(f"[SAVE_FAIL] path={tuple_output_json_path} err={e}")

    return results


def _parse_bf_cf_captions(captions_text: str, n_initial: int, n_detail: int) -> (List[Optional[str]], List[Optional[str]]):
    bf_caps: List[Optional[str]] = [None] * max(0, n_initial)
    cf_caps: List[Optional[str]] = [None] * max(0, n_detail)
    bf_unassigned: List[str] = []
    cf_unassigned: List[str] = []
    try:
        if not captions_text:
            return bf_caps, cf_caps
        # 支持 BVF/BF 与 CVF/CF，大小写、可选空格、英文/中文冒号
        pat_bvf = re.compile(r"^\s*(?:bvf|bf)\s*([0-9]+)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)
        pat_cvf = re.compile(r"^\s*(?:cvf|cf)\s*([0-9]+)\s*[:：]\s*(.+?)\s*$", re.IGNORECASE)
        for raw in captions_text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = pat_bvf.match(line)
            if m:
                try:
                    idx_num = int(m.group(1))
                    cap = m.group(2).strip()
                    # 同时兼容 0/1 基索引
                    if 0 <= idx_num < n_initial:
                        # 若位置已被占，用顺序队列备用
                        if bf_caps[idx_num] is None:
                            bf_caps[idx_num] = cap
                        else:
                            bf_unassigned.append(cap)
                    elif 1 <= idx_num <= n_initial:
                        pos = idx_num - 1
                        if bf_caps[pos] is None:
                            bf_caps[pos] = cap
                        else:
                            bf_unassigned.append(cap)
                except Exception:
                    pass
                continue
            m = pat_cvf.match(line)
            if m:
                try:
                    idx_num = int(m.group(1))
                    cap = m.group(2).strip()
                    if 0 <= idx_num < n_detail:
                        if cf_caps[idx_num] is None:
                            cf_caps[idx_num] = cap
                        else:
                            cf_unassigned.append(cap)
                    elif 1 <= idx_num <= n_detail:
                        pos = idx_num - 1
                        if cf_caps[pos] is None:
                            cf_caps[pos] = cap
                        else:
                            cf_unassigned.append(cap)
                except Exception:
                    pass
                continue
        # 用顺序补齐未填位置，保证无 None
        for i in range(len(bf_caps)):
            if bf_caps[i] is None and bf_unassigned:
                bf_caps[i] = bf_unassigned.pop(0)
        for j in range(len(cf_caps)):
            if cf_caps[j] is None and cf_unassigned:
                cf_caps[j] = cf_unassigned.pop(0)
        # 仍为空的置为空串，避免 None
        for i in range(len(bf_caps)):
            if bf_caps[i] is None:
                bf_caps[i] = ""
        for j in range(len(cf_caps)):
            if cf_caps[j] is None:
                cf_caps[j] = ""
    except Exception:
        pass
    return bf_caps, cf_caps


def generate_experience_from_captions(
    question_text: str,
    step_idx_text: str,
    selected_l0_index: str,
    selected_l1_index: str,
    n_initial: int,
    n_detail: int,
    captions_text: str,
    initial_rels: List[str],
    detail_rels: List[str],
    final_reward: str,
) -> Optional[str]:
    """
    Use the produced captions as the primary textual evidence to write ONE paragraph (12–16 sentences),
    explicitly covering the two-stage structure and answering the critique points.
    """
    outcome = None
    if final_reward in {"pass", "fail"}:
        outcome = "PASS" if final_reward == "pass" else "FAIL"

    # Critique prompt (system): background and requirements per user's template
    sys_prompt = (
        "You are an embodied agent located in an indoor environment and can perceive the environment with a camera. "
        "You are given a question and you need to explore the environment to answer the question. "
        "At each step, you should choose a frontier where to move in the next step to solve the task as efficiently as possible. "
        "At each step, you are firstly given a list of 'broad-view frontiers' (BVF), which are coarse exploration directions that cover distinct areas of the scene, and you select the index of one BVF to look closer. "
        "Under the selected BVF, you are provided with a list of 'closer-view frontiers' (CVF) to choose from as your destination for the next step. CVF are detailed snapshots of the selected BVF’s direction. "
        "You are given enough context of when you selected a frontier at a step; your current task is to generate a critique about your decision of frontier selection. Before the critique, begin with a short descriptive prelude: briefly characterize what the broader-view set looked like and explicitly state which broader view was chosen and why; then briefly characterize the closer-view set under that broader view and explicitly state which closer view was chosen and why(together in 2-3 sentences). After this prelude, address the critique. It should cover the following aspects: "
        "whether the choice aimed to explore unseen area or solve the task immediately; how the current frontier selection potentially impacts the exploration in the next step; how the timing of this choice in the ongoing process shaped its impact on the final outcome; how the current frontier selection affects the final outcome of the task; whether there was any better selection among alternative frontiers and why. "
        "After the critique, generate one-sentence experience abstraction you learn from this exploration trial. This experience abstraction is focused on how you can solve similar task or explore in similar environment in the future. The experience abstraction should be generalisable and transferable for your exploration of future tasks"
        "RULE: Your output must be ONE single paragraph (12–16 sentences), past tense, with concrete and precise narration. Begin with the exact words 'Critique: In this step,' then briefly paraphrase the current question, and continue concisely. Refer only to objects and visual evidence; avoid symbolic labels or indices. \n\n"
        "Output in the following format:\n"
        "Critique: <one paragraph>\n"
        "Abstraction: <one sentence summary instruction>"
    )
    # generate abstraction again. 


    # User content per template
    content = []
    total_steps = 50
    content.append((f"The current question you are solving is {question_text}.",))
    content.append((f"And now you are at step {step_idx_text} out of {total_steps} step.",))

    # Parse captions then render BVF/CVF lines
    bf_caps, cf_caps = _parse_bf_cf_captions(captions_text or "", n_initial, n_detail)
    content.append(("You are given the following BVFs and CVFs:",))
    bvf_lines: List[str] = []
    for i in range(n_initial):
        cap = bf_caps[i] if i < len(bf_caps) and bf_caps[i] else ""
        bvf_lines.append(f"BVF{i+1}: {cap}")
    if bvf_lines:
        content.append(("\n\n".join(bvf_lines),))
    cvf_lines: List[str] = []
    for j in range(n_detail):
        cap = cf_caps[j] if j < len(cf_caps) and cf_caps[j] else ""
        cvf_lines.append(f"CVF{j+1}: {cap}")
    if cvf_lines:
        content.append(("\n\n".join(cvf_lines),))

    content.append((f"You have chosen BVF {selected_l0_index} to look closer, and chosen CVF {selected_l1_index} to explore in the next step.",))
    if outcome:
        content.append((f"This task has {outcome} in the end.",))
    content.append(("Please generate a critique about your decision of frontier selection in the current step.",))

    return call_openai_api(sys_prompt, content)


    

def _img_file_to_b64(abs_path: str) -> Optional[str]:
    if not abs_path or not os.path.exists(abs_path):
        return None
    try:
        with open(abs_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        return None


def _ensure_searcher(output_parent_dir: str, exp_name: str, questions_list_path: Optional[str], replay_json_path: str) -> Optional[FrontierSimilaritySearcher]:
    if not os.path.exists(replay_json_path):
        logging.error(f"Input JSON not found: {replay_json_path}")
        return None
    return FrontierSimilaritySearcher(output_parent_dir, exp_name, questions_list_path, replay_json_path, method="ahash")


def generate_experiences(
    input_json_path: str,
    output_json_path: str,
    output_parent_dir: str,
    exp_name: str,
    questions_list_path: Optional[str],
) -> Dict[str, Any]:
    """
    Generate one-to-one experience text for each frontier image based on the
    input JSON (with structure similar to replay_step_info.json).

    Output structure example:
    {
      episode_id: {
        question_id: {
          "steps": {
            step_key: {
              "experience": {
                "layer0": { "frontier/<l0_file>": "..." },
                "layer1": { "frontier/<l1_file>": "..." }
              }
            }
          }
        }
      }
    }
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    searcher = _ensure_searcher(output_parent_dir, exp_name, questions_list_path, input_json_path)
    if searcher is None:
        raise FileNotFoundError(f"Cannot create searcher, input JSON not found: {input_json_path}")

    # Load existing output for resume (do not overwrite processed steps)
    results: Dict[str, Any] = {}
    if output_json_path and os.path.exists(output_json_path):
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                results = json.load(f) or {}
            logging.info(f"[Resume] Loaded existing output: {output_json_path}")
        except Exception as e:
            logging.warning(f"[Resume] Failed to load existing output, starting fresh. err={e}")

    base_root = os.path.join(output_parent_dir, exp_name)

    # Iterate episodes -> questions -> steps
    for episode_id, qdict in data.items():
        if not isinstance(qdict, dict):
            continue
        results.setdefault(episode_id, {})

        for question_id, qinfo in qdict.items():
            if not isinstance(qinfo, dict):
                continue
            results[episode_id].setdefault(question_id, {})

            steps = qinfo.get("steps", {}) or {}
            if not isinstance(steps, dict):
                continue

            results[episode_id][question_id].setdefault("steps", {})

            # Exclude the current question during search
            exclude_qid = question_id

            for step_key, step_info in steps.items():
                if not isinstance(step_info, dict):
                    continue

                frontier = step_info.get("frontier", {}) or {}
                if not isinstance(frontier, dict):
                    frontier = {}

                # Collect layer0 keys and all associated layer1 relative paths
                layer0_keys = list(frontier.keys())

                # Skip if this step already has a non-empty experience in existing results
                existing_step = (
                    results.get(episode_id, {})
                           .get(question_id, {})
                           .get("steps", {})
                           .get(step_key, {})
                           .get("experience")
                )
                if isinstance(existing_step, str) and existing_step.strip():
                    logging.info(f"[Resume] Skip processed step: {episode_id}/{question_id}/{step_key}")
                    continue

                # Output container (single unified experience per step)
                step_out = {
                    "experience": None,
                }

                # Generate captions first, then generate final experience using captions
                try:
                    chosen_frontier = step_info.get("chosen_frontier", {}) or {}
                    chosen_l1 = chosen_frontier.get("layer1")
                    chosen_l0 = chosen_frontier.get("layer0")

                    anchor_rel = chosen_l1 or chosen_l0
                    anchor_level = "layer1" if chosen_l1 else "layer0"

                    # step index parsed from key like 'step_0'
                    try:
                        _idx = int(str(step_key).split("_")[-1])
                    except Exception:
                        _idx = None

                    # Build broad (initial) and closer lists used for captioning
                    initial_keys = list(frontier.keys())
                    initial_rels = [os.path.join("frontier", k) for k in initial_keys]
                    # Apply same fallback correction as in captions-only path
                    chosen_l0_eff = chosen_l0
                    chosen_l1_eff = chosen_l1
                    initial_key_for_details = None
                    if chosen_l0_eff and isinstance(chosen_l0_eff, str) and chosen_l0_eff.startswith("frontier/"):
                        initial_key_for_details = chosen_l0_eff.split("/", 1)[1]
                    detail_rels = frontier.get(initial_key_for_details, []) if initial_key_for_details else []
                    if chosen_l1_eff and not detail_rels:
                        l1_to_k = {}
                        all_l1 = []
                        for k in initial_keys:
                            for rel in (frontier.get(k, []) or []):
                                l1_to_k[rel] = k
                                all_l1.append(rel)
                        best_l1 = _best_match_l1_rel(chosen_l1_eff, all_l1)
                        if best_l1 and best_l1 in l1_to_k:
                            owning_k = l1_to_k[best_l1]
                            detail_rels = frontier.get(owning_k, []) or []
                            chosen_l0_eff = os.path.join("frontier", owning_k)
                    if chosen_l0_eff and chosen_l0_eff not in initial_rels:
                        alt_l0 = _best_match_l0_rel(chosen_l0_eff, initial_rels)
                        if alt_l0:
                            chosen_l0_eff = alt_l0
                            initial_key_for_details = chosen_l0_eff.split("/", 1)[1]
                            detail_rels = frontier.get(initial_key_for_details, []) or detail_rels
                    if chosen_l1_eff and detail_rels and (chosen_l1_eff not in detail_rels):
                        alt_l1 = _best_match_l1_rel(chosen_l1_eff, detail_rels)
                        if alt_l1:
                            chosen_l1_eff = alt_l1

                    # Indices (0-based) for selected items (string for prompt)
                    def _idx0s_l0(rel):
                        p = _parse_bvf_idx_from_rel(rel)
                        if p is not None:
                            return str(p)
                        try:
                            return str(initial_rels.index(rel)) if rel and rel in initial_rels else "NA"
                        except Exception:
                            return "NA"
                    def _idx0s_l1(rel):
                        p = _parse_cvf_idx_from_rel(rel)
                        if p is not None:
                            return str(p)
                        try:
                            return str(detail_rels.index(rel)) if rel and rel in detail_rels else "NA"
                        except Exception:
                            return "NA"

                    # 不允许 NA：若 NA 则从文件名解析或默认 0
                    def _sanitize_idx_s(val: str, rel: Optional[str], parse_fn) -> str:
                        if val != "NA":
                            return val
                        try:
                            p = parse_fn(rel)
                            if isinstance(p, int):
                                return str(max(0, p))
                        except Exception:
                            pass
                        return "0"

                    selected_l0_index = _sanitize_idx_s(_idx0s_l0(chosen_l0 or chosen_l0_eff), chosen_l0 or chosen_l0_eff, _parse_bvf_idx_from_rel)
                    selected_l1_index = _sanitize_idx_s(_idx0s_l1(chosen_l1 or chosen_l1_eff), chosen_l1 or chosen_l1_eff, _parse_cvf_idx_from_rel)

                    # Step index text
                    step_idx_text = "NA" if _idx is None else str(max(0, min(49, _idx)))

                    # Question text and final reward for this question
                    question_text = (qinfo.get("question", "") or "").strip()
                    final_reward = (qinfo.get("final_reward", None) or "").strip().lower()

                    # Stage A: caption all candidates (bf then cf)
                    captions_text = generate_captions_for_frontiers(
                        searcher=searcher,
                        question_id=question_id,
                        question_text=question_text,
                        step_idx_text=step_idx_text,
                        initial_rels=initial_rels,
                        detail_rels=detail_rels,
                    )

                    # Stage B: critique-based experience using captions
                    t = generate_experience_from_captions(
                        question_text=question_text,
                        step_idx_text=step_idx_text,
                        selected_l0_index=selected_l0_index,
                        selected_l1_index=selected_l1_index,
                        n_initial=len(initial_rels),
                        n_detail=len(detail_rels),
                        captions_text=captions_text or "",
                        initial_rels=initial_rels,
                        detail_rels=detail_rels,
                        final_reward=final_reward,
                    )
                    if t:
                        step_out["experience"] = t.strip()
                        logging.info(f"[Experience] {episode_id}/{question_id}/{step_key}: {step_out['experience']}")
                except Exception as e:
                    logging.warning(f"step-level context failed: epi={episode_id} q={question_id} step={step_key} err={e}")

                # Merge into results and write incrementally to disk

                results[episode_id].setdefault(question_id, {}).setdefault("steps", {})[step_key] = step_out
                try:
                    if output_json_path:
                        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
                        with open(output_json_path, "w", encoding="utf-8") as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        logging.info(f"[Save] Wrote progress to {output_json_path}")
                except Exception as e:
                    logging.warning(f"[Save] Failed to write progress: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch-generate experience text for each frontier from input JSON and write to an output JSON")
    parser.add_argument("--input_json", required=True, help="Path to input JSON (structure similar to replay_step_info.json)")
    parser.add_argument("--output_json", required=True, help="Path to output experience JSON")
    parser.add_argument("--output_parent_dir", required=True, help="Output root directory (contains experiment subdir)")
    parser.add_argument("--exp_name", required=True, help="Experiment name (can include nested subdirs)")
    parser.add_argument("--questions_list_path", default=None, help="Optional questions list JSON path")
    parser.add_argument("--captions_only", action="store_true", help="Only generate captions and write exp_tuple.json aligned with given experience JSON")
    parser.add_argument(
        "--experience_json_path",
        default="/anvme/workspace/v100dd12-3dmem/openeqa/ee_qwen/qwen-exp-168/experience_output.json",
        help="Path to reference experience_output.json for parsing Critique/Abstraction and alignment",
    )
    parser.add_argument("--tuple_output_json", default=None, help="Output path for exp_tuple.json (defaults to dirname(output_json)/exp_tuple.json)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.captions_only:
        tuple_output = args.tuple_output_json
        if not tuple_output:
            tuple_output = os.path.join(os.path.dirname(os.path.abspath(args.output_json)), "exp_tuple.json")

        logging.info("[Mode] Captions-only. Generating captions and writing tuples JSON.")
        results = generate_caption_tuples(
            input_json_path=args.input_json,
            output_parent_dir=args.output_parent_dir,
            exp_name=args.exp_name,
            experience_json_path=args.experience_json_path,
            tuple_output_json_path=tuple_output,
        )
        print(tuple_output)
    else:
        results = generate_experiences(
            input_json_path=args.input_json,
            output_json_path=args.output_json,
            output_parent_dir=args.output_parent_dir,
            exp_name=args.exp_name,
            questions_list_path=args.questions_list_path,
        )
        print(args.output_json)


if __name__ == "__main__":
    main()


