import os
import json
import argparse
import logging
from typing import Dict, List, Optional, Tuple

# Reuse the existing OpenAI chat wrapper
from src.eval_utils_gpt_aeqa_qwen import call_openai_api




def format_abstraction_prompt(*args, **kwargs):
    # Deprecated: kept for compatibility in case of external imports. Not used in this script.
    raise NotImplementedError("Deprecated in favor of two-stage pipeline (captions->critiques->abstraction).")


def build_abstraction_for_question(*args, **kwargs):
    # Deprecated: kept for compatibility in case of external imports. Not used in this script.
    raise NotImplementedError("Deprecated in favor of two-stage pipeline (captions->critiques->abstraction).")







def format_final_trajectory_abstraction_prompt(
    question_text: str,
    segments: List[str],
    task_outcome: Optional[str] = None,
) -> Tuple[str, List[Tuple[str, str]]]:

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


def generate_traj_abstraction_from_chunk_caption(
    chunk_caption_data: dict,
    question_id: str,
    seed: Optional[int] = None,
) -> Optional[dict]:
    """
    Generate abstraction directly from chunk caption data.
    
    Args:
        chunk_caption_data: dict containing question_text, caption, and status
        question_id: the question ID
        seed: optional random seed
        
    Returns a dict: {
      'question': str,
      'abstraction': str,
    }
    or None if the data is invalid.
    """
    if not isinstance(chunk_caption_data, dict):
        return None
        
    question_text = chunk_caption_data.get("question_text", "")
    caption = chunk_caption_data.get("caption", "")
    status = chunk_caption_data.get("status", "")
    
    if not question_text or not caption:
        return None

    # Use the caption directly as the trajectory description
    # Generate final abstraction using the caption
    final_sys, final_cont = format_final_trajectory_abstraction_prompt(
        question_text,
        segments=[caption],  # Use the caption as a single segment
        task_outcome=("PASS" if status.lower() == "pass" else ("FAIL" if status.lower() == "fail" else None))
    )
    final_out = call_openai_api(final_sys, final_cont, seed=seed)
    final_out = (final_out or "").strip()
    _, abstraction_text = _split_thinking_and_abstraction(final_out)
    logging.info(f"[Traj][Final] qid={question_id} final_abs_len={len(abstraction_text)}")

    return {
        "question": question_text,
        "abstraction": abstraction_text,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a generalized Abstraction from chunk caption data."
    )
    parser.add_argument(
        "--chunk_caption",
        type=str,
        required=True,
        help="Path to chunk caption json file containing question_text, caption, and status.",
    )
    parser.add_argument(
        "--question_id",
        type=str,
        required=False,
        default=None,
        help="Target question_id to process. Omit for processing all.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional chat seed (overrides env VLLM_SEED).",
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
    return parser.parse_args()


def load_chunk_caption_data(chunk_caption_path: str) -> Dict[str, dict]:
    """
    Load chunk caption data from JSON file.
    Returns mapping {question_id: {question_text, caption, status}}.
    """
    if not os.path.exists(chunk_caption_path):
        logging.warning(f"chunk_caption file not found: {chunk_caption_path}")
        return {}
    try:
        with open(chunk_caption_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return data
    except Exception as e:
        logging.error(f"failed to read chunk_caption: {e}")
        return {}


def main():
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)

    # Load chunk caption data
    chunk_data = load_chunk_caption_data(args.chunk_caption)
    if not chunk_data:
        logging.warning("no chunk caption data found; nothing to do")
        return

    # Mode 1: single question
    if isinstance(args.question_id, str) and len(args.question_id) > 0:
        if args.question_id not in chunk_data:
            logging.warning(f"question_id {args.question_id} not found in chunk data")
            print("{}")
            return
            
        result_obj = generate_traj_abstraction_from_chunk_caption(
            chunk_caption_data=chunk_data[args.question_id],
            question_id=args.question_id,
            seed=args.seed,
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
    if not isinstance(args.out, str) or len(args.out) == 0:
        raise SystemExit("--out is required when processing all questions")

    qids = sorted(chunk_data.keys())
    if isinstance(args.max_questions, int) and args.max_questions > 0:
        qids = qids[: args.max_questions]

    results: Dict[str, dict] = {}
    for qid in qids:
        try:
            obj = generate_traj_abstraction_from_chunk_caption(
                chunk_caption_data=chunk_data[qid],
                question_id=qid,
                seed=args.seed,
            )
            if obj is None:
                results[qid] = {"question": "", "abstraction": ""}
            else:
                results[qid] = obj
            # incremental write after each question
            _write_incremental_json(args.out, qid, results[qid])
        except Exception as e:
            logging.warning(f"abstraction generation failed for {qid}: {e}")
            results[qid] = {"question": "", "abstraction": ""}
            _write_incremental_json(args.out, qid, results[qid])

    # final write to ensure consistency
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"[Write] wrote batch results for {len(results)} questions to {args.out}")


if __name__ == "__main__":
    main()


# def format_final_trajectory_abstraction_prompt(
    # question_text: str,
    # segments: List[str],
    # task_outcome: Optional[str] = None,
    # ) -> Tuple[str, List[Tuple[str, str]]]:
    # """
    # Generate a final abstraction preserving full step-by-step reasoning (Step 0–4),
    # producing two complementary outputs:
    # (1) Environment Dynamics — environment map & functionality abstraction.
    # (2) Decision-making Skills — task-oriented operational workflow.
    # Each block must contain 10–12 sentences of concrete, condition-based rules.
    # """

    # sys_prompt = (
    #     "You are to synthesize a final trajectory-level abstraction for an embodied exploration agent. "
    #     "INPUT: several trajectory paragraphs describing exploration for the SAME question. "
    #     "You MUST perform INTERNAL reasoning through FIVE steps (Step 0–4) as defined below. "
    #     "These steps guide your thinking but MUST NOT appear in your visible output.\n\n"

    #     "=== INTERNAL THINKING STEPS (DO NOT PRINT) ===\n"
    #     "Step 0 (Task Understanding): define the question type, target entity, success criteria, "
    #     "evidence channels, verification method, and stop conditions.\n"
    #     "Step 1 (Trajectory Synthesis): reconstruct spatial path logic, key transitions, evidence-driven moves, and coverage logic.\n"
    #     "Step 2 (Env–Object Associations): infer stable mappings between regions and object categories; identify functional zones and cues.\n"
    #     "Step 3 (Strategy × Question Type + Directional Priors): derive condition–action tactics per question type, pivot/stop rules, and verification routines.\n"
    #     "Step 4 (Anti-patterns): identify concrete failure modes and formulate counter-rules.\n\n"

    #     "=== VISIBLE OUTPUT ONLY ===\n"
    #     "You must output EXACTLY TWO labeled paragraphs with the following intent:\n\n"

    #     "Environment Dynamics:\n"
    #     "- 10–12 sentences.\n"
    #     "- Present the environment as a *functional map* for navigation: describe major regions, their typical functions, "
    #     "what can be found or done there, and how regions connect.\n"
    #     "- Each sentence should express concrete, environment-grounded knowledge (e.g., 'entrances often contain signage and containers'; "
    #     "'utility corners near sinks hold cleaning tools'; 'corridors connect living spaces to service zones').\n"
    #     "- Emphasize spatial logic, region affordances, and distribution of cues, not immediate task execution.\n"
    #     "- Think of this section as describing *what the world offers* and *where key evidence tends to appear*.\n\n"

    #     "Decision-making Skills:\n"
    #     "- 10–12 sentences.\n"
    #     "- Present the general *operational workflow* for performing the task type defined in Step 0.\n"
    #     "- Each sentence should encode actionable, conditional behavior ('when X, do Y; if no cue, pivot to Z; avoid ...').\n"
    #     "- Integrate directional priors, pivot/stop rules, and local verification tactics.\n"
    #     "- If Task Outcome is FAIL, embed 2–4 explicit 'do-not/avoid' sentences indicating concrete missteps to prevent.\n"
    #     "- Think of this section as describing *how to act and decide* given the environment described above.\n\n"

    #     "GLOBAL RULES (apply to both):\n"
    #     "- NEVER mention 'BVF', 'CVF', 'view', 'snapshot', 'image', camera operations, or step IDs.\n"
    #     "- Use only region/landmark/path words (entrance, corridor, connector, hub, doorway, kitchen zone, utility corner, counter, board, panel).\n"
    #     "- Use imperative or conditional tone; avoid summaries like 'lessons learned' or 'this highlights the importance of...'.\n"
    #     "- Respect the 10–12 sentence range strictly for each block.\n"
    # )

    # content: List[Tuple[str, str]] = []
    # content.append((f"Question: {question_text or '(unknown)'}",))
    # if isinstance(task_outcome, str) and task_outcome.strip():
    #     content.append((f"Task Outcome: {task_outcome.strip().upper()}",))
    # content.append(("Segments:",))
    # for i, seg in enumerate(segments, start=1):
    #     if isinstance(seg, str) and seg.strip():
    #         content.append((f"Segment {i}: {seg.strip()}",))

    # content.append((
    #     "Now, perform INTERNAL reasoning through Steps 0–4 (as defined above). "
    #     "After finishing your reasoning, output exactly two labeled paragraphs:\n\n"
    #     "Environment Dynamics: A single 10–12 sentence paragraph describing the environment as a functional and navigable map — "
    #     "what zones exist, what functions or cues they provide, and how they interconnect.\n\n"
    #     "Decision-making Skills: A single 10–12 sentence paragraph describing the general operational workflow — "
    #     "conditional tactics, pivot/stop rules, verification patterns, and avoidance behaviors (include 2–4 'do-not/avoid' rules if FAIL)."
    # ,))

    # return sys_prompt, content