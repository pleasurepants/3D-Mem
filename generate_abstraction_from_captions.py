import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple

# Reuse the existing OpenAI chat wrapper
from src.eval_utils_gpt_aeqa_qwen import call_openai_api


def _load_tuple_node(exp_tuple_path: str, question_id: str) -> Optional[dict]:
    """
    Load the question node for a given question_id from an exp_tuple json file.
    Compatible with two structures:
      1) Top-level mapping: { question_id: {"question": str, "step_0": {...}, ...} }
      2) Top-level episodes: { episode_key: { question_id: {"question": str, "steps": {...}} } }
    """
    if not exp_tuple_path or not os.path.exists(exp_tuple_path):
        logging.warning(f"exp_tuple file not found: {exp_tuple_path}")
        return None
    try:
        with open(exp_tuple_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"failed to read exp_tuple: {e}")
        return None

    if not isinstance(data, dict):
        return None

    # Structure 1
    if question_id in data and isinstance(data[question_id], dict):
        return data[question_id]

    # Structure 2
    for _, bucket in data.items():
        if isinstance(bucket, dict) and question_id in bucket and isinstance(bucket[question_id], dict):
            return bucket[question_id]

    return None


def _collect_steps(qnode: dict) -> Tuple[str, List[Tuple[str, dict]]]:
    """
    Return the question text and a sorted list of (step_key, step_dict).
    Supports both scattered step_* and nested steps dict.
    """
    if not isinstance(qnode, dict):
        return "", []

    question_text = qnode.get("question", "")

    steps_obj = None
    if "steps" in qnode and isinstance(qnode["steps"], dict):
        steps_obj = qnode["steps"]
    else:
        steps_obj = {k: v for k, v in qnode.items() if isinstance(v, dict) and k.startswith("step_")}

    if not steps_obj:
        return question_text, []

    def _step_order(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 0

    items = sorted(list(steps_obj.items()), key=lambda kv: _step_order(kv[0]))
    return question_text, items


def _truncate(text: Optional[str], max_chars: int = 600) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[: max(0, max_chars - 3)] + "..."


def _is_question_node(node: dict) -> bool:
    if not isinstance(node, dict):
        return False
    if not isinstance(node.get("question"), str):
        return False
    if isinstance(node.get("steps"), dict):
        return True
    for k, v in node.items():
        if isinstance(v, dict) and isinstance(k, str) and k.startswith("step_"):
            return True
    return False


def load_all_question_nodes(exp_tuple_path: str) -> Dict[str, dict]:
    """
    Load all question nodes from the tuple file, supporting both top-level qid and
    episode->qid layouts. Returns mapping {question_id: question_node}.
    """
    result: Dict[str, dict] = {}
    if not exp_tuple_path or not os.path.exists(exp_tuple_path):
        logging.warning(f"exp_tuple file not found: {exp_tuple_path}")
        return result
    try:
        with open(exp_tuple_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logging.error(f"failed to read exp_tuple: {e}")
        return result

    if not isinstance(data, dict):
        return result

    # Case A: top-level qid
    for k, v in data.items():
        if isinstance(k, str) and _is_question_node(v):
            result[k] = v

    # Case B: episode -> qid
    for _, bucket in data.items():
        if isinstance(bucket, dict):
            for qid, qnode in bucket.items():
                if isinstance(qid, str) and _is_question_node(qnode):
                    result[qid] = qnode

    return result


def format_abstraction_prompt(*args, **kwargs):
    # Deprecated: kept for compatibility in case of external imports. Not used in this script.
    raise NotImplementedError("Deprecated in favor of two-stage pipeline (captions->critiques->abstraction).")


def build_abstraction_for_question(*args, **kwargs):
    # Deprecated: kept for compatibility in case of external imports. Not used in this script.
    raise NotImplementedError("Deprecated in favor of two-stage pipeline (captions->critiques->abstraction).")


def format_grouped_critique_from_step_critiques_prompt(
    question_text: str,
    critiques_by_steps: List[Tuple[str, str]],  # list of (step_key, critique_text)
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Summarize a group of existing step-level Critiques into ONE consolidated Critique paragraph.
    Output must be a single line starting with 'Critique:'.
    """
    sys_prompt = (
        "You are consolidating step-level critiques for an embodied exploration agent. "
        "You will be given a question and multiple critiques, each evaluating a single step's frontier decision (what was chosen, why, what it led to, and its effect on progress/outcome). "
        "Your task is to synthesize these critiques into ONE consolidated critique that captures recurring decision patterns, strengths, mistakes, missed opportunities, and their impact on progress and final outcome. Treat the policy of choosing a broader direction and then a closer view as a fixed baseline (not up for debate); avoid restating or judging that baseline. Focus on whether the chosen path progressed toward the goal. "
        "Begin with a brief prelude (2–3 sentences) analyzing the exploration trajectory across these steps based on the prior choices (as a sequence of decisions). "
        "Then, critique this trajectory segment by focusing on: whether the sequence prioritized exploration vs immediate answering; how the sequence of selections affected subsequent exploration; how the timing within this trajectory segment shaped impact; how this sequence influenced the final outcome; and whether better alternatives for this segment existed and why. "
    )
    sys_prompt += (
        "\nRULE: Your output must be ONE single paragraph (12–16 sentences), past tense, precise and objective. "
        "Begin with the exact words 'Critique:' and continue concisely. Base your writing only on the provided critiques; do not invent new details. Avoid symbolic indices and do not mention internal sub-choice labels. \n\n"
        "Output in the following format:\n"
        "Critique: <one paragraph>"
    )

    content: List[Tuple[str, str]] = []
    content.append((f"Question: {question_text or '(unknown)'}",))
    content.append(("Below are the step critiques to review:",))
    for step_key, ct in critiques_by_steps:
        if isinstance(ct, str) and ct.strip():
            content.append((f"{step_key}: {ct.strip()}",))
    content.append(("Now produce the consolidated single-line critique as specified.",))
    return sys_prompt, content


def format_abstraction_from_critiques_prompt(
    question_text: str,
    critiques: List[str],
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Build a prompt that aggregates multiple critiques into ONE trajectory-level Abstraction.
    Output must be a single line starting with 'Abstraction:'.
    """
    sys_prompt = (
        "You are to synthesize trajectory-level guidance (traj-abstraction) for an embodied exploration agent. "
        "INPUT: a question and multiple critiques; each critique is itself a condensed evaluation of multiple steps within a short trajectory segment (what was chosen, why, what it led to, and its effect on progress/outcome). "
        "GOAL: produce decision-useful, generalizable guidance that applies across similar scenes and tasks. Prefer category/region words and cues over instance-specific nouns. Avoid empty slogans and avoid step IDs or internal labels. Do NOT mention or discuss 'BVF', 'CVF', 'view', 'snapshot', 'image', or camera operations—focus on strategy and environment–object/task mappings only.\n\n"
        "OUTPUT FORMAT (print all steps explicitly, then the final one-paragraph abstraction):\n"
        "Step 1 (Trajectory): 2–4 sentences summarizing the exploration trajectory as a decision sequence for THIS question (no IDs; do not mention views).\n"
        "Step 2 (Env–Object Associations): 1–2 sentences distilling where typical categories are likely found (generalized; e.g., storage/cleaning near utility areas; signage near entrances/hubs).\n"
        "Step 3 (Strategy × Question Type): 1–2 sentences giving concrete, non-generic guidance per question type (location: use region priors to shortlist areas; attribute/state: prioritize proximity checks of the target category using functional cues; counting/relationship: first gain coverage to enumerate, then verify local relations; text-reading: seek signage/labels/panels).\n"
        "Step 4 (Directional Priors & Avoidance): 1–2 sentences on which directions/cues tend to help vs derail (e.g., connectors/hubs vs dead-end clutter; signage-bearing corridors vs closed, textureless corners).\n"
        "Step 5 (Anti-patterns): 1–2 sentences describing common failure modes to avoid for similar tasks (e.g., fixating on decor or tool clutter when the question targets containers/appliances; roaming without leveraging region priors).\n"
        "FINAL Abstraction: Start the line with 'Abstraction: ' and then VERBATIM concatenate all sentences you printed in Steps 1–5 into a single paragraph (keep the exact wording; do not add, remove, or paraphrase any words; do not introduce new content; do not mention views/BVF/CVF)."
    )

    content: List[Tuple[str, str]] = []
    content.append((f"Question: {question_text or '(unknown)'}",))
    content.append(("Here are the step-level condensed critiques to synthesize:",))
    for i, c in enumerate(critiques, start=1):
        if isinstance(c, str) and c.strip():
            content.append((f"Critique {i}: {c.strip()}",))
    content.append((
        "Now print the steps exactly in the specified order (Step 1 .. Step 5), and then print FINAL Abstraction by copying (verbatim) all sentences from Steps 1–5 into one paragraph prefixed with 'Abstraction: '. Do not use step IDs or internal labels.",
    ))
    return sys_prompt, content


def format_trajectory_from_captions_prompt(
    question_text: str,
    captions_by_steps: List[Tuple[str, str, Optional[int], Optional[int]]],  # (step_key, caption, chosen_bvf, chosen_cvf)
    task_outcome: Optional[str] = None,
) -> Tuple[str, List[Tuple[str, str]]]:

    sys_prompt = (
        "You are summarizing a movement trajectory in an indoor environment from step captions. "
        "BASELINE POLICY: the agent first selects a broader direction (BVF) and then a closer sub-direction (CVF) within that BVF. This two-stage selection is a fixed baseline and is NOT to be debated; use the provided chosen BVF/CVF only as internal evidence to understand movement direction. "
        "INPUT: a question and several step captions (what was visible/focused at each step), together with the chosen BVF and CVF indices at each step. "
        "TASK: condense these steps into a clear sequence of where the agent moved or focused in the environment (trajectory), using region/landmark words (e.g., hallway, entrance, kitchen zone, sink area). Do NOT mention or discuss 'BVF', 'CVF', 'view', 'snapshot', 'image', or camera operations in your outputs; avoid step IDs; use natural language. "
        "After summarizing the trajectory, write a Critique paragraph reflecting on the route per the given questions. Finally, print an Abstraction line that repeats the Captions paragraph verbatim."
    )
    sys_prompt += (
        "\nOUTPUT (print exactly in this order):\n"
        "Captions: <ONE single paragraph with 16–20 sentences summarizing the overall trajectory and movement logic derived from the captions; use only region/landmark and path terms; avoid any mention of views/BVF/CVF/images; avoid step IDs; cohesive, non-bulleted prose>\n"
        "Critique: <ONE paragraph addressing: whether choices aimed to explore unseen vs solve immediately; how selections impacted the next steps; how timing in the trajectory shaped impact; how selections influenced final outcome; whether better alternatives existed and why>"
    )

    content: List[Tuple[str, str]] = []
    content.append((f"Question: {question_text or '(unknown)'}",))
    if isinstance(task_outcome, str) and task_outcome.strip():
        content.append((f"Task Outcome: {task_outcome.strip().upper()}",))
    content.append(("Below are the step captions and the chosen indices for each step (for your internal reasoning; do NOT mention BVF/CVF in outputs):",))
    for step_key, cap, bvf, cvf in captions_by_steps:
        if isinstance(cap, str) and cap.strip():
            content.append((f"{step_key} Caption: {cap.strip()}",))
        content.append((f"{step_key} Chosen: BVF {('NA' if bvf is None else str(bvf))}; CVF {('NA' if cvf is None else str(cvf))}",))
    content.append((
        "Now print EXACTLY two blocks in order: (1) Captions paragraph (16–20 sentences), (2) Critique paragraph. Then print Abstraction: <repeat the Captions paragraph verbatim>. Do not add any other headers or lines.",
    ))
    return sys_prompt, content


def format_final_trajectory_abstraction_prompt(
    question_text: str,
    segments: List[str],
    task_outcome: Optional[str] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """
    基于多个分段（由前一阶段从 captions 汇总得到的段落）做 CoT 风格整合：先按步骤思考，再输出最终 Abstraction。
    输出格式（严格）：
      Step 1 (Trajectory): 2–4 句，总结整个问题的全局轨迹与段落之间的衔接（仅区域/地标/路径用词，不含视角/BVF/CVF/图片/ID）
      Step 2 (Env–Object Associations): 1–2 句，总结关键类别与区域的关联先验
      Step 3 (Strategy × Question Type): 1–2 句，给出不同问题类型的策略匹配（位置/属性-状态/计数-关系/读文本）
      Step 4 (Directional Priors & Avoidance): 1–2 句，指出哪些方向/线索常有帮助/常无效
      Step 5 (Anti-patterns): 1–2 句，指出常见失败模式
      Abstraction: 以 'Abstraction: ' 开头的一段话（16–20 句），把以上要点整合为可执行、可迁移的总体指导；无需也不得出现视角/BVF/CVF/图片/ID
    """
    sys_prompt = (
        "You are to synthesize a final trajectory-level abstraction for an embodied exploration agent. "
        "INPUT: several trajectory paragraphs, each summarizing a short segment for the SAME question. "
        "TASK: think step by step to integrate these segments, then output steps and a final Abstraction. "
        "STRICT FORMAT: Your output MUST contain EXACTLY SIX blocks in this order and with these labels: \n"
        "Step 0 (Task Understanding) — 2–3 sentences\n"
        "Step 1 (Trajectory) — 8–10 sentences\n"
        "Step 2 (Env–Object Associations) — 4–6 sentences\n"
        "Step 3 (Strategy × Question Type + Directional Priors) — 4–6 sentences\n"
        "Step 4 (Anti-patterns) — 2–3 sentences\n"
        "Abstraction: <20–24 sentence cohesive paragraph>\n"
        "No extra lines, no additional headers, and do NOT reorder or omit any block. Do NOT mention 'BVF', 'CVF', 'view', 'snapshot', 'image', camera operations, or step IDs anywhere. Use only region/landmark/path terms and task-relevant cues. "
        "In the Abstraction paragraph, always include concrete environment-task priors (e.g., 'recycling stations near utility/kitchen zones', 'signage near entrances/hubs'). If Task Outcome is FAIL, also include 2–4 explicit lessons phrased as do-not/avoid rules (e.g., 'avoid lingering in cluttered corners without new cues', 'do not switch directions without fresh evidence')."
    )
    content: List[Tuple[str, str]] = []
    content.append((f"Question: {question_text or '(unknown)'}",))
    if isinstance(task_outcome, str) and task_outcome.strip():
        content.append((f"Task Outcome: {task_outcome.strip().upper()}",))
    content.append(("Segments:",))
    for i, seg in enumerate(segments, start=1):
        if isinstance(seg, str) and seg.strip():
            content.append((f"Segment {i}: {seg.strip()}",))
    content.append((
        "Now print the following blocks in the exact order and with the exact labels (no extra content before/after). Constraints for ALL blocks: do NOT mention BVF/CVF/views/snapshots/images/camera; do NOT use step IDs; use only region/landmark/path words and task-relevant cues; keep prose, no bullets.\n\n"
        "Step 0 (Task Understanding) — 2–3 sentences: Paraphrase succinctly what the question asks (e.g., find/verify/compare), and what constitutes success.\n\n"
        "Step 1 (Trajectory) — 8–10 sentences: Summarize the overall trajectory across segments.\n"
        "- Describe the entry points, major regions/rooms traversed (e.g., entrance, hallway, kitchen zone, utility area, living space), and key transitions between them.\n"
        "- Indicate movement directionality (toward/away from salient regions or landmarks) and why the route changed (e.g., encountering new evidence or exhausting an area).\n"
        "- Focus on path logic and coverage (what was visited first/next/last), not on per-image details.\n\n"
        "Step 2 (Env–Object Associations) — 4–6 sentences: General priors linking categories to regions.\n"
        "- Use generic categories and regions (e.g., signage near entrances/hubs; cookware in kitchen-like areas; cleaning supplies near sinks/utility corners; clothing/linens near bedroom/closet zones).\n"
        "- Avoid scene-specific item names.\n\n"
        "Step 3 (Strategy × Question Type + Directional Priors) — 4–6 sentences: Concrete guidance per question type with directional priors.\n"
        "- Location: shortlist regions via priors, then confirm in the most indicative sub-areas.\n"
        "- Attribute/State: prioritize proximity checks of the target category using functional/visual cues; verify state locally.\n"
        "- Counting/Relationship: gain coverage to enumerate instances first, then verify local relations.\n"
        "- Text-reading: seek text-bearing surfaces/signage/panels with high-contrast lettering near decision points (entrances, hubs, boards).\n"
        "- Helpful: connectors (hallways/intersections), doorways, hubs; Harmful: blind dead-ends, purely cluttered corners without new cues.\n\n"
        "Step 4 (Anti-patterns) — 2–3 sentences: Common failure modes to avoid.\n"
        "- Make it concrete and environment-aware: specify where/when NOT to go. For example: following the perimeter of closed garage doors yields little new evidence when searching for containers; diving into deep storage alcoves is unhelpful for text-reading tasks; lingering in decor-heavy corners seldom helps container/appliance queries; circling vehicle bays rarely reveals recycling signage. Also state when to stop: avoid repeating passes along blank walls or returning to dead-end utility closets after container zones were already scanned; do not switch directions without fresh evidence; treat wrong or full-bin findings as negative evidence to pivot early.\n\n"
        "**Abstraction**: <20–24 sentence cohesive paragraph integrating Steps 1–5 into actionable, transferable guidance for similar tasks. Do not introduce scope beyond Steps 1–5; do not mention BVF/CVF/views/images; do not use step IDs.>",
    ))
    return sys_prompt, content


def _extract_answer(qnode: dict) -> str:
    # Try multiple common keys; fallback to empty string
    for k in ["answer", "final_answer", "gt_answer", "ground_truth", "pred_answer"]:
        v = qnode.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _split_thinking_and_abstraction(full_text: Optional[str]) -> Tuple[str, str]:
    """
    Split model output into (thinking_process, abstraction_only).
    - thinking_process: everything before the first line starting with 'Abstraction:' or '**Abstraction**:' (case-sensitive for double-asterisk; case-insensitive for the rest handled by explicit checks)
    - abstraction_only: the content after the marker on that line (trimmed), plus any subsequent lines
    If no marker is found, returns (full_text or '', '').
    """
    if not isinstance(full_text, str) or not full_text.strip():
        return "", ""
    lines = [ln.rstrip() for ln in full_text.splitlines()]
    abs_idx = -1
    # Preferred marker: **Abstraction**: (with colon)
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s.startswith("**Abstraction**:"):
            abs_idx = i
            break
    # Accept **Abstraction** (no colon)
    if abs_idx == -1:
        for i, ln in enumerate(lines):
            if ln.strip() == "**Abstraction**":
                abs_idx = i
                break
    # Fallback to plain Abstraction: (with colon, case-insensitive)
    if abs_idx == -1:
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith("abstraction:"):
                abs_idx = i
                break
    # Accept plain Abstraction (no colon), as a title line
    if abs_idx == -1:
        for i, ln in enumerate(lines):
            if ln.strip().lower() == "abstraction":
                abs_idx = i
                break
    # If still not found, treat entire text as abstraction
    if abs_idx == -1:
        return "", full_text.strip()
    thinking = "\n".join(lines[:abs_idx]).strip()
    abs_line = lines[abs_idx]
    # Take content after the first colon on the marker line (if any). If no colon, drop the title.
    if ":" in abs_line:
        first_part = abs_line.split(":", 1)[1].lstrip()
    else:
        first_part = ""
    tail = "\n".join(lines[abs_idx + 1:]).strip()
    abstraction_only = (first_part + ("\n" + tail if tail else "")).strip()
    return thinking, abstraction_only


def _write_incremental_json(out_path: str, qid: str, obj: dict):
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if os.path.exists(out_path):
            try:
                with open(out_path, "r", encoding="utf-8") as rf:
                    existing = json.load(rf) or {}
            except Exception:
                existing = {}
        else:
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        existing[qid] = obj
        with open(out_path, "w", encoding="utf-8") as wf:
            json.dump(existing, wf, ensure_ascii=False, indent=2)
        logging.info(f"[Write][incremental] wrote qid={qid} to {out_path}")
    except Exception as e:
        logging.warning(f"[Write][incremental] failed for qid={qid} path={out_path}: {e}")


def generate_traj_abstraction_with_critiques(
    exp_tuple_path: str,
    question_id: str,
    seed: Optional[int] = None,
    max_steps: Optional[int] = None,
) -> Optional[dict]:
    """
    Two-stage pipeline per question:
      1) Group step critiques by 'group size' (this function uses max_steps as group size) and generate one Critique per group.
      2) Aggregate all group critiques into a single trajectory-level Abstraction.

    Returns a dict: {
      'question': str,
      'answer': str,
      'abstraction': str,           # 'Abstraction: ...'
      'critiques': { 'step_0_4': 'Critique: ...', ... }
    }
    or None if the question_id is not found.
    """
    qnode = _load_tuple_node(exp_tuple_path, question_id)
    if not isinstance(qnode, dict):
        return None

    question_text, steps = _collect_steps(qnode)
    if not steps:
        return None

    # Prepare groups of step-level captions
    step_pairs: List[Tuple[str, str, Optional[int], Optional[int]]] = []  # (step_key, caption, bvf, cvf)
    for step_key, sd in steps:
        cap = sd.get("Caption") or sd.get("caption") or ""
        bvf = sd.get("chosen_BVF") if isinstance(sd.get("chosen_BVF"), int) else None
        cvf = sd.get("chosen_CVF") if isinstance(sd.get("chosen_CVF"), int) else None
        step_pairs.append((step_key, (cap if isinstance(cap, str) else ""), bvf, cvf))

    # Group into chunks by group size (use max_steps as grouping size)
    trajectories_combined: List[str] = []  # collect Abstraction-only per chunk
    thinking_all: List[str] = []          # collect Step lines per chunk
    critiques_map: Dict[str, str] = {}    # kept for compatibility; now unused
    group_size = max_steps if isinstance(max_steps, int) and max_steps > 0 else 5
    for i in range(0, len(step_pairs), group_size):
        chunk = step_pairs[i : i + group_size]
        logging.info(f"[Traj] qid={question_id} chunk={i//group_size+1} range_keys={[kp for kp,_1,_2,_3 in chunk]}")
        sys_p, cont = format_trajectory_from_captions_prompt(
            question_text, chunk,
            task_outcome=("PASS" if (str(qnode.get("final_reward", "")).lower() == "pass") else ("FAIL" if (str(qnode.get("final_reward", "")).lower() == "fail") else None))
        )
        traj_out = call_openai_api(sys_p, cont, seed=seed)
        traj_text = (traj_out or "").strip()
        # store by range key like step_0_4 (kept for compatibility)
        start_key = chunk[0][0]
        end_key = chunk[-1][0]
        try:
            start_idx = int(start_key.split("_")[-1])
        except Exception:
            start_idx = i
        try:
            end_idx = int(end_key.split("_")[-1])
        except Exception:
            end_idx = i + len(chunk) - 1
        range_key = f"step_{start_idx}_{end_idx}"
        # split thinking vs abstraction
        thinking_process, abstr_only = _split_thinking_and_abstraction(traj_text)
        if thinking_process:
            thinking_all.append(thinking_process)
        if abstr_only:
            trajectories_combined.append(abstr_only)
        logging.info(f"[Traj] qid={question_id} chunk={i//group_size+1} abstr_len={len(abstr_only)} tp_len={len(thinking_process)}")
        critiques_map[range_key] = traj_text  # Optional: raw per-chunk record

    # Final abstraction: summarize across chunk-level paragraphs into one final paragraph via one more VLM call
    final_sys, final_cont = format_final_trajectory_abstraction_prompt(
        question_text,
        segments=trajectories_combined,
        task_outcome=("PASS" if (str(qnode.get("final_reward", "")).lower() == "pass") else ("FAIL" if (str(qnode.get("final_reward", "")).lower() == "fail") else None))
    )
    final_out = call_openai_api(final_sys, final_cont, seed=seed)
    final_out = (final_out or "").strip()
    thinking_process = "\n".join([t for t in thinking_all if t])
    _, abstraction_text = _split_thinking_and_abstraction(final_out)
    logging.info(f"[Traj][Final] qid={question_id} segments={len(trajectories_combined)} final_abs_len={len(abstraction_text)} tp_total_len={len(thinking_process)}")

    return {
        "question": question_text,
        "abstraction": abstraction_text,
        "thinking_process": thinking_process,
        "captions_step": critiques_map,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a generalized Abstraction for a question_id by summarizing its step logs."
    )
    parser.add_argument(
        "--exp_tuple",
        type=str,
        required=True,
        help="Path to exp_tuple json (no default; pass via shell).",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        required=False,
        default=None,
        help="Target question_id to summarize. Omit for processing all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional chat seed (overrides env VLLM_SEED).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optionally cap the number of steps included (use earliest).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output json file. For single question, writes {qid: text}. For batch, writes all.",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=None,
        help="Optional cap when processing all questions.",
    )
    # removed: group_size; we always consolidate all selected steps into one critique
    return parser.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    # Mode 1: single question
    if isinstance(args.question_id, str) and len(args.question_id) > 0:
        result_obj = generate_traj_abstraction_with_critiques(
            exp_tuple_path=args.exp_tuple,
            question_id=args.question_id,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        if result_obj is None:
            print("{}")
            return
        print(json.dumps({args.question_id: result_obj}, ensure_ascii=False, indent=2))
        if isinstance(args.out, str) and len(args.out) > 0:
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump({args.question_id: result_obj}, f, ensure_ascii=False, indent=2)
            logging.info(f"[Write] wrote single question result to {args.out}")
        return

    # Mode 2: process all questions in the file
    all_nodes = load_all_question_nodes(args.exp_tuple)
    if not all_nodes:
        logging.warning("no question nodes found in exp_tuple; nothing to do")
        return
    if not isinstance(args.out, str) or len(args.out) == 0:
        raise SystemExit("--out is required when processing all questions")

    qids = sorted(all_nodes.keys())
    if isinstance(args.max_questions, int) and args.max_questions > 0:
        qids = qids[: args.max_questions]

    results: Dict[str, dict] = {}
    for qid in qids:
        try:
            obj = generate_traj_abstraction_with_critiques(
                exp_tuple_path=args.exp_tuple,
                question_id=qid,
                seed=args.seed,
                max_steps=args.max_steps,
            )
            if obj is None:
                results[qid] = {"question": "", "answer": "", "abstraction": "", "critiques": {}}
            else:
                results[qid] = obj
            # incremental write after each question
            _write_incremental_json(args.out, qid, results[qid])
        except Exception as e:
            logging.warning(f"abstraction generation failed for {qid}: {e}")
            results[qid] = {"question": "", "answer": "", "abstraction": "", "critiques": {}}
            _write_incremental_json(args.out, qid, results[qid])

    # final write to ensure consistency
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"[Write] wrote batch results for {len(results)} questions to {args.out}")


if __name__ == "__main__":
    main()


