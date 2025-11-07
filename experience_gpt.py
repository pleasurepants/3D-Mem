import os
import json
import re
import argparse
import logging
from typing import Dict, List, Optional, Tuple
# Reuse the existing OpenAI chat wrapper
from src.eval_utils_gpt_aeqa_qwen import call_openai_api
import openai
from openai import OpenAI
import time
from src.const_gpt import *
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


def format_abstraction_prompt(*args, **kwargs):
    # Deprecated: kept for compatibility in case of external imports. Not used in this script.
    raise NotImplementedError("Deprecated in favor of two-stage pipeline (captions->critiques->abstraction).")


def build_abstraction_for_question(*args, **kwargs):
    # Deprecated: kept for compatibility in case of external imports. Not used in this script.
    raise NotImplementedError("Deprecated in favor of two-stage pipeline (captions->critiques->abstraction).")


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
                model="gpt-4o",  # gpt-4o-internvl-minicpm-qwen
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



# v-1
# def format_final_trajectory_abstraction_prompt(
#     question_text: str,
#     segments: List[str],
#     task_outcome: Optional[str] = None,
# ) -> Tuple[str, List[Tuple[str, str]]]:

#     sys_prompt = (
#         "You are to synthesize a final trajectory-level abstraction for an embodied exploration agent. "
#         "INPUT: several trajectory paragraphs, each summarizing a short segment for the SAME question. "
#         "TASK: think step by step to integrate these segments, then output steps and a final Abstraction. "
#         "STRICT FORMAT: Your output MUST contain EXACTLY SIX blocks in this order and with these labels: \n"
#         "Step 0 (Task Understanding) — 2–3 sentences\n"
#         "Step 1 (Trajectory) — 8–10 sentences\n"
#         "Step 2 (Env–Object Associations) — 4–6 sentences\n"
#         "Step 3 (Strategy × Question Type + Directional Priors) — 4–6 sentences\n"
#         "Step 4 (Anti-patterns) — 2–3 sentences\n"
#         "Abstraction: <20–24 sentence cohesive paragraph>\n"
#         "No extra lines, no additional headers, and do NOT reorder or omit any block. Do NOT mention 'BVF', 'CVF', 'view', 'snapshot', 'image', camera operations, or step IDs anywhere. Use only region/landmark/path terms and task-relevant cues. "
#         "In the Abstraction paragraph, always include concrete environment-task priors (e.g., 'recycling stations near utility/kitchen zones', 'signage near entrances/hubs'). If Task Outcome is FAIL, also include 2–4 explicit lessons phrased as do-not/avoid rules (e.g., 'avoid lingering in cluttered corners without new cues', 'do not switch directions without fresh evidence')."
#     )
#     content: List[Tuple[str, str]] = []
#     content.append((f"Question: {question_text or '(unknown)'}",))
#     if isinstance(task_outcome, str) and task_outcome.strip():
#         content.append((f"Task Outcome: {task_outcome.strip().upper()}",))
#     content.append(("Segments:",))
#     for i, seg in enumerate(segments, start=1):
#         if isinstance(seg, str) and seg.strip():
#             content.append((f"Segment {i}: {seg.strip()}",))
#     content.append((
#         "Now print the following blocks in the exact order and with the exact labels (no extra content before/after). Constraints for ALL blocks: do NOT mention BVF/CVF/views/snapshots/images/camera; do NOT use step IDs; use only region/landmark/path words and task-relevant cues; keep prose, no bullets.\n\n"
#         "Step 0 (Task Understanding) — 2–3 sentences: Paraphrase succinctly what the question asks (e.g., find/verify/compare), and what constitutes success.\n\n"
#         "Step 1 (Trajectory) — 8–10 sentences: Summarize the overall trajectory across segments.\n"
#         "- Describe the entry points, major regions/rooms traversed (e.g., entrance, hallway, kitchen zone, utility area, living space), and key transitions between them.\n"
#         "- Indicate movement directionality (toward/away from salient regions or landmarks) and why the route changed (e.g., encountering new evidence or exhausting an area).\n"
#         "- Focus on path logic and coverage (what was visited first/next/last), not on per-image details.\n\n"
#         "Step 2 (Env–Object Associations) — 4–6 sentences: General priors linking categories to regions.\n"
#         "- Use generic categories and regions (e.g., signage near entrances/hubs; cookware in kitchen-like areas; cleaning supplies near sinks/utility corners; clothing/linens near bedroom/closet zones).\n"
#         "- Avoid scene-specific item names.\n\n"
#         "Step 3 (Strategy × Question Type + Directional Priors) — 4–6 sentences: Concrete guidance per question type with directional priors.\n"
#         "- Location: shortlist regions via priors, then confirm in the most indicative sub-areas.\n"
#         "- Attribute/State: prioritize proximity checks of the target category using functional/visual cues; verify state locally.\n"
#         "- Counting/Relationship: gain coverage to enumerate instances first, then verify local relations.\n"
#         "- Text-reading: seek text-bearing surfaces/signage/panels with high-contrast lettering near decision points (entrances, hubs, boards).\n"
#         "- Helpful: connectors (hallways/intersections), doorways, hubs; Harmful: blind dead-ends, purely cluttered corners without new cues.\n\n"
#         "Step 4 (Anti-patterns) — 2–3 sentences: Common failure modes to avoid.\n"
#         "- Make it concrete and environment-aware: specify where/when NOT to go. For example: following the perimeter of closed garage doors yields little new evidence when searching for containers; diving into deep storage alcoves is unhelpful for text-reading tasks; lingering in decor-heavy corners seldom helps container/appliance queries; circling vehicle bays rarely reveals recycling signage. Also state when to stop: avoid repeating passes along blank walls or returning to dead-end utility closets after container zones were already scanned; do not switch directions without fresh evidence; treat wrong or full-bin findings as negative evidence to pivot early.\n\n"
#         "**Abstraction**: <20–24 sentence cohesive paragraph integrating Steps 1–5 into actionable, transferable guidance for similar tasks. Do not introduce scope beyond Steps 1–5; do not mention BVF/CVF/views/images; do not use step IDs.>",
#     ))
#     return sys_prompt, content




# v0
# def format_final_trajectory_abstraction_prompt(
#     question_text: str,
#     segments: List[str],
#     task_outcome: Optional[str] = None,
# ) -> Tuple[str, List[Tuple[str, str]]]:

#     # sys_prompt: global rules plus enforced CoT structure (defines format without embedding specific inputs)
#     sys_prompt = (
#         "You are a self-reflective embodied exploration agent.\n"
#         "Your task is to produce a two-part analysis of a full exploration trajectory with a STRICT step-by-step Chain-of-Thought style in <reflection>.\n\n"
#         "=== INPUT SCHEMA (provided separately) ===\n"
#         "<Target task>...</Target task>\n"
#         "<exploration trajectory>...</exploration trajectory>\n"
#         "<Final outcome>...</Final outcome>\n\n"
#         "=== OUTPUT FORMAT (must be exact; HTML-like tags; no extra headers or sections) ===\n"
#         "<reflection>\n"
#         "  <step1_task_understanding></step1_task_understanding>\n"
#         "  <step2_trajectory_reconstruction></step2_trajectory_reconstruction>\n"
#         "  <step3_strategy_balance></step3_strategy_balance>\n"
#         "  <step4_directional_shifts></step4_directional_shifts>\n"
#         "  <step5_phase_timing></step5_phase_timing>\n"
#         "  <step6_style_analysis></step6_style_analysis>\n"
#         "  <step7_alternative_global_strategy></step7_alternative_global_strategy>\n"
#         "  <summary_explain_outcome></summary_explain_outcome>\n"
#         "</reflection>\n\n"
#         "<abstraction>\n"
#         "</abstraction>\n\n"
#         "=== REFLECTION (step-by-step content requirements) ===\n"
#         "- <step1_task_understanding>: State the overarching goal and success condition succinctly.\n"
#         "- <step2_trajectory_reconstruction>: Reconstruct the overall route: which regions came first vs. later, transitions, and reasons for changes.\n"
#         "- <step3_strategy_balance>: Judge if the trajectory prioritized broad coverage vs. immediate problem-solving; explain how that balance affected efficiency/success.\n"
#         "- <step4_directional_shifts>: Identify major directional/region shifts; which improved evidence gain, which caused stagnation, and why.\n"
#         "- <step5_phase_timing>: Explain how early/mid/late sequencing and timing shaped the final outcome.\n"
#         "- <step6_style_analysis>: Describe how local choices aggregated into a general exploration style; name systemic strengths/weaknesses.\n"
#         "- <step7_alternative_global_strategy>: Propose a better global strategy (if any), specifying how ordering, regions, or pivots would change and why it would likely improve performance.\n"
#         "- <summary_explain_outcome>: Tie the above analysis to the final outcome (SUCCESS/FAIL) and list 2–3 concrete improvements for a redo.\n\n"
#         "=== ABSTRACTION (writing guidelines) ===\n"
#         "- The <abstraction> block must contain exactly FOUR labeled parts in this order and format:\n"
#         "  General: <one cohesive paragraph with 3–5 sentences of transferable reasoning rules.>\n"
#         "  Specific: <one cohesive paragraph with 5–7 sentences detailing environment–task priors and concrete cues.>\n"
#         "  Positive Lessons: <2–4 sentences highlighting what to prioritize or repeat in future similar tasks.>\n"
#         "  Negative Lessons: <2–4 sentences identifying pitfalls or strategies to avoid next time.>\n"
#         "- Each part must start with its label exactly as shown (no extra tags or markdown). Keep the full <abstraction> block coherent and self-contained (14–18 sentences total).\n"
#         "— Positive/Negative specificity rules —\n"
#         "- In 'Positive Lessons', write 2–4 if–then rules that each include: (i) a concrete region or landmark, (ii) a perceptual/functional cue, (iii) an action verb, and (iv) an expected effect tied to <reflection> (reference which step influenced it, e.g., 'from step3/step5').\n"
#         "- In 'Negative Lessons', write 2–4 anti-pattern rules that each include: (i) a trigger condition, (ii) a stop/pivot criterion with a small numeric threshold (e.g., 'within 1–2 transitions'), and (iii) a counterfactual fix linked to <reflection> (e.g., 'pivot to utility corners because of early-phase timing in step5').\n"
#         "- Every sentence in 'Positive Lessons' and 'Negative Lessons' must contain at least one region/landmark term and one concrete action; avoid generic terms like 'be efficient', 'avoid detours' unless paired with a condition and a remedy.\n"
#         "- Prefer measurable phrasing (e.g., 'after scanning two non-informative rooms', 'within two doorway transitions', 'when signage density is low, switch to high-prior zones').\n"

#         "=== CONSTRAINTS ===\n"
#         "- Use only regions/landmarks/paths and task-relevant cues. Do NOT mention cameras, images, BVF/CVF, or step IDs beyond the required tags.\n"
#         "- Keep tag names and order EXACTLY as specified. No extra sections, bullets, or markdown headers. "
#         "In <abstraction>, extend and consolidate the reasoning from <reflection>, reusing its concrete insights and environment–task priors instead of introducing new generalities. "
#         "Give special weight to <summary_explain_outcome> when writing <abstraction>, expanding its causal reasoning and implications into a longer, detailed synthesis of about 14–18 sentences."
#         " The <abstraction> block must be written as plain prose only—no sub-tags such as <general>, <specific>, <positive_lessons>, or <negative_lessons> are allowed."
#         "It must comprehensively integrate the reasoning and conclusions from all steps in <reflection>—covering task understanding, trajectory logic, strategy balance, directional shifts, phase timing, exploration style, and alternative strategies—into one cohesive synthesis."


#     )

#     # content: inject concrete inputs (HTML blocks) and supply minimal instructions that enforce the required format
#     content: List[Tuple[str, str]] = []

#     content.append((
#         "<Target task>\n"
#         f"{question_text or '(unknown)'}\n"
#         "</Target task>",
#     ))

#     if segments:
#         joined_segments = []
#         for i, seg in enumerate(segments, start=1):
#             if isinstance(seg, str) and seg.strip():
#                 joined_segments.append(f"#chunk {i}: {seg.strip()}")
#         trajectory_text = "\n".join(joined_segments) if joined_segments else "(no segments)"
#     else:
#         trajectory_text = "(no segments)"

#     content.append((
#         "<exploration trajectory>\n"
#         f"{trajectory_text}\n"
#         "</exploration trajectory>",
#     ))

#     final_outcome_str = (task_outcome or "").strip() or "(unknown)"
#     content.append((
#         "<Final outcome>\n"
#         f"{final_outcome_str}\n"
#         "</Final outcome>",
#     ))

#     content.append((
#         "Now produce the output strictly in the required format with the exact tags and order:\n"
#         "<reflection>\n"
#         "  <step1_task_understanding>...</step1_task_understanding>\n"
#         "  <step2_trajectory_reconstruction>...</step2_trajectory_reconstruction>\n"
#         "  <step3_strategy_balance>...</step3_strategy_balance>\n"
#         "  <step4_directional_shifts>...</step4_directional_shifts>\n"
#         "  <step5_phase_timing>...</step5_phase_timing>\n"
#         "  <step6_style_analysis>...</step6_style_analysis>\n"
#         "  <step7_alternative_global_strategy>...</step7_alternative_global_strategy>\n"
#         "  <summary_explain_outcome>...</summary_explain_outcome>\n"
#         "</reflection>\n\n"
#         "In <abstraction>, extend from <reflection> by reusing its concrete insights and environment–task priors; do not introduce new generalities.\n"
#         "Make 'Positive Lessons' and 'Negative Lessons' concrete: use if–then / anti-pattern + pivot rules with regions, cues, actions, numeric thresholds\n"
#         "<abstraction>...</abstraction>\n"
#     ))

#     return sys_prompt, content



# v1
def format_final_trajectory_abstraction_prompt(
    question_text: str,
    segments: List[str],
    task_outcome: Optional[str] = None,
    exp_mode: str = "lessons",
) -> Tuple[str, List[Tuple[str, str]]]:

    if exp_mode == "unformat":
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

    sys_prompt = (
        "You are a self-reflective embodied exploration agent.\n"
        "Your task is to produce a two-part analysis of a full exploration trajectory with a STRICT step-by-step Chain-of-Thought style in REFLECTION.\n\n"
        "=== INPUT SCHEMA (provided separately) ===\n"
        "<Target task>...</Target task>\n"
        "<exploration trajectory>...</exploration trajectory>\n"
        "<Final outcome>...</Final outcome>\n\n"
        "=== OUTPUT FORMAT (must be exact; plain-text labels; no extra headers or sections) ===\n"
        "REFLECTION:\n"
        "task_understanding: ...\n"
        "trajectory_reconstruction: ...\n"
        "strategy_balance: ...\n"
        "directional_shifts: ...\n"
        "phase_timing: ...\n"
        "style_assessment: ...\n"
        "alternative_strategy: ...\n"
        "outcome_synthesis: ...\n\n"
        "ABSTRACTION:\n"
        "Positive Lessons: ...\n"
        "Negative Lessons: ...\n\n"
        "=== REFLECTION (analysis requirements) ===\n"
        "- task_understanding: Restate the exact guiding question verbatim (quote question_text) while outlining the overarching goal and success condition.\n"
        "- trajectory_reconstruction: Reconstruct the overall route: which regions came first vs. later, transitions, and reasons for changes—all in service of answering question_text.\n"
        "- strategy_balance: Judge if the trajectory prioritized broad coverage vs. immediate problem-solving; explain how that balance aligned with the question's requirements and affected efficiency/success.\n"
        "- directional_shifts: Identify major directional/region shifts; which improved evidence gain, which caused stagnation, and why, citing how each shift related to question_text.\n"
        "- phase_timing: Explain how early/mid/late sequencing and timing shaped the final outcome relative to the question's demands.\n"
        "- style_assessment: Describe how local choices aggregated into an overall exploration style; name systemic strengths/weaknesses with respect to the question.\n"
        "- alternative_strategy: Propose a better global strategy (if any), specifying how ordering, regions, or pivots would change and why it would likely improve performance for this question.\n"
        "- outcome_synthesis: Tie the above analysis to the final outcome (SUCCESS/FAIL), explicitly link back to the guiding question's success criteria (include the question wording or its key nouns), and list 2–3 concrete improvements for a redo.\n"
        "- Throughout REFLECTION, do not reference numeric step labels; instead, cite evidence or facets of question_text, weaving its targets, actions, and environments into every subsection.\n\n"
        "=== ABSTRACTION (writing guidelines) ===\n"
        "- The ABSTRACTION section must contain exactly TWO labeled parts in this order and format:\n"
        "  Positive Lessons: <one paragraph with 5–6 sentences highlighting what to prioritize or repeat in future similar tasks. Begin with a single sentence that restates the guiding question and task goal, quoting question_text or a faithful paraphrase.>\n"
        "  Negative Lessons: <one paragraph with 5–6 sentences identifying pitfalls or strategies to avoid next time. Begin with a single sentence that frames the key risks or failure modes for the same question, again citing question_text or its core terms.>\n"
        "- Apply the General/Specific/Concise principles as writing constraints: every sentence must express a transferable rule, anchor it in concrete regions/landmarks/cues/timing markers, and stay compact and action-oriented.\n"
        "- Ensure both paragraphs explicitly reference how the question framing or answer criteria shaped the recommended moves, without naming step numbers.\n"
        "- After the opening sentence in each paragraph, every subsequent sentence must directly reuse concrete findings from REFLECTION (e.g., specific regions visited, directional pivots, timing judgments, style assessments, or outcome explanations) and explicitly mention a question-specific element (target object, required action, success cue).\n"
        "- Keep the full ABSTRACTION section coherent and self-contained (10–12 sentences total, across both paragraphs).\n"
        "- In 'Positive Lessons', write 2–4 if–then rules that each include: (i) a concrete region or landmark, (ii) a perceptual or functional cue, (iii) an action verb, and (iv) an expected effect tied to the REFLECTION analysis (refer to evidence or findings rather than step numbers).\n"
        "- In 'Negative Lessons', write 2–4 anti-pattern rules that each include: (i) a trigger condition, (ii) a stop/pivot criterion with a small numeric threshold (e.g., 'within 1–2 transitions'), and (iii) a counterfactual fix linked to the REFLECTION analysis (cite observations instead of step numbers).\n"
        "- Every sentence in both paragraphs must include at least one region/landmark term and one concrete action; avoid generic phrasing and keep measurements explicit (e.g., 'after scanning two non-informative rooms', 'within two doorway transitions').\n"

        "=== CONSTRAINTS ===\n"
        "- Use only regions/landmarks/paths and task-relevant cues. Do NOT mention cameras, images, BVF/CVF, or step IDs beyond the required labels.\n"
        "- Keep every label EXACTLY as specified and in order. No extra sections, bullets, or markdown headers. "
        "In ABSTRACTION, extend and consolidate the reasoning from REFLECTION, reusing its concrete insights and environment–task priors instead of introducing new umbrella themes. "
        "Give special weight to summary_explain_outcome when writing ABSTRACTION, expanding its causal reasoning, question-specific implications, and directional lessons into a focused synthesis of about 10–12 sentences."
        " Make sure every ABSTRACTION statement clearly signals which REFLECTION insight it extends or compresses (e.g., by mirroring its terminology or paraphrasing its causal link)."
        " The ABSTRACTION section must be written as plain prose only—no additional sub-headings beyond the two required labels are allowed."
        "It must comprehensively integrate the reasoning and conclusions from all REFLECTION steps—covering task understanding, trajectory logic, strategy balance, directional shifts, phase timing, exploration style, and alternative strategies—into one cohesive synthesis."


    )

    # content: inject concrete inputs (HTML blocks) and supply minimal trigger instructions that enforce the required format
    content: List[Tuple[str, str]] = []

    content.append((
        "<Target task>\n"
        f"{question_text or '(unknown)'}\n"
        "</Target task>",
    ))

    if segments:
        joined_segments = []
        for i, seg in enumerate(segments, start=1):
            if isinstance(seg, str) and seg.strip():
                joined_segments.append(f"#chunk {i}: {seg.strip()}")
        trajectory_text = "\n".join(joined_segments) if joined_segments else "(no segments)"
    else:
        trajectory_text = "(no segments)"

    content.append((
        "<exploration trajectory>\n"
        f"{trajectory_text}\n"
        "</exploration trajectory>",
    ))

    final_outcome_str = (task_outcome or "").strip() or "(unknown)"
    content.append((
        "<Final outcome>\n"
        f"{final_outcome_str}\n"
        "</Final outcome>",
    ))

    content.append((
        "Now produce the output strictly in the required format with the exact labels and order:\n"
        "REFLECTION:\n"
        "step1_task_understanding: ...\n"
        "step2_trajectory_reconstruction: ...\n"
        "step3_strategy_balance: ...\n"
        "step4_directional_shifts: ...\n"
        "step5_phase_timing: ...\n"
        "step6_style_analysis: ...\n"
        "step7_alternative_global_strategy: ...\n"
        "summary_explain_outcome: ...\n\n"
        "ABSTRACTION:\n"
        "Positive Lessons: ...\n"
        "Negative Lessons: ...\n"
        "\n"
        "In ABSTRACTION, extend from REFLECTION by reusing its concrete insights and environment–task priors; do not introduce new umbrella themes.\n"
        "Only include the two labeled parts above; each must be a single paragraph with 5–6 sentences.\n"
        "Quote the literal question_text in the REFLECTION task_understanding subsection and in the opening sentences of both ABSTRACTION paragraphs.\n"
        "Write every sentence as a transferable yet concrete rule anchored in regions/landmarks/cues/timing markers, using if–then or anti-pattern guidance with measurable thresholds, and explicitly connect recommendations back to the guiding question.\n"
        "After the opening sentence, ensure every ABSTRACTION sentence references a question-specific element (target object, required action, or success cue) and mirrors a specific REFLECTION insight.\n"
        "Keep the overall ABSTRACTION section within 10–12 sentences across the two paragraphs.\n"
        "Each ABSTRACTION sentence must mirror or paraphrase a specific REFLECTION insight so the two sections stay tightly coupled.\n"
        "Do not reference step numbers anywhere in the output; ground reasoning in the question, evidence, and environment cues instead.\n"
    ))

    return sys_prompt, content







def _split_thinking_and_abstraction(full_text: Optional[str]) -> Tuple[str, str]:
    """
    Split model output into (reflection, abstraction).
    Supports the current plain-text label format and older HTML-like/tagged formats.
    Falls back to legacy parsing (looking for 'Abstraction:' markers) when labels are missing.
    """
    if not isinstance(full_text, str) or not full_text.strip():
        return "", ""
    
    # Try the plain-text label format first (REFLECTION / ABSTRACTION)
    reflection_plain = re.search(
        r'REFLECTION:\s*(.*?)(?=\n\s*ABSTRACTION:)',
        full_text,
        re.DOTALL | re.IGNORECASE,
    )
    abstraction_plain = re.search(r'ABSTRACTION:\s*(.*)', full_text, re.DOTALL | re.IGNORECASE)
    if reflection_plain and abstraction_plain:
        reflection_content = reflection_plain.group(1).strip()
        abstraction_content = abstraction_plain.group(1).strip()
        return reflection_content, abstraction_content
    
    # Fallback: try HTML-like tag format
    reflection_match = re.search(r'<reflection>(.*?)</reflection>', full_text, re.DOTALL | re.IGNORECASE)
    abstraction_match = re.search(r'<abstraction>(.*?)</abstraction>', full_text, re.DOTALL | re.IGNORECASE)
    
    if reflection_match and abstraction_match:
        reflection_content = reflection_match.group(1).strip()
        abstraction_content = abstraction_match.group(1).strip()
        return reflection_content, abstraction_content
    
    # Fallback to legacy parsing for backward compatibility
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
    # Accept heading formats with markdown prefixes (e.g., ### Abstraction)
    if abs_idx == -1:
        for i, ln in enumerate(lines):
            normalized = re.sub(r'^[#*\s]+', '', ln.strip())
            normalized = re.sub(r'\*\*', '', normalized)
            lower_norm = normalized.lower()
            if lower_norm in {"abstraction", "abstraction:"} or lower_norm.startswith("abstraction:"):
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


def _empty_result_for_mode(mode: str) -> Dict[str, object]:
    if mode == "unformat":
        return {
            "question": "",
            "abstraction": "",
            "thinking_process": "",
            "captions_step": {},
        }
    return {
        "question": "",
        "reflection": "",
        "abstraction": "",
    }


def generate_traj_abstraction_from_chunk_caption(
    chunk_caption_data: dict,
    question_id: str,
    seed: Optional[int] = None,
    exp_mode: str = "lessons",
) -> Optional[dict]:
    """
    Generate abstraction directly from chunk caption data.
    
    Args:
        chunk_caption_data: dict containing question_text, caption, and status
        question_id: the question ID
        seed: optional random seed
        
    Returns a dict: {
      'question': str,
      'reflection': str,
      'abstraction': str,
    }
    or None if the data is invalid.
    """
    if not isinstance(chunk_caption_data, dict):
        return None
    
    mode = (exp_mode or "lessons").lower()
    if mode not in {"lessons", "unformat"}:
        mode = "lessons"

    question_text = (
        chunk_caption_data.get("question_text")
        or chunk_caption_data.get("question")
        or ""
    )
    caption = (
        chunk_caption_data.get("caption")
        or chunk_caption_data.get("trajectory")
        or chunk_caption_data.get("experience")
        or chunk_caption_data.get("traj")
        or ""
    )
    status = chunk_caption_data.get("status", "") or chunk_caption_data.get("final_reward", "")
    
    if not question_text or not caption:
        return None

    # Use the caption (or provided trajectory) directly as the trajectory description
    # Generate final abstraction using the selected prompt style
    final_sys, final_cont = format_final_trajectory_abstraction_prompt(
        question_text,
        segments=[caption],  # Treat the provided text as a single segment
        task_outcome=("PASS" if status.lower() == "pass" else ("FAIL" if status.lower() == "fail" else None)),
        exp_mode=mode,
    )
    final_out = call_openai_api(final_sys, final_cont, seed=seed)
    final_out = (final_out or "").strip()
    thinking_text, abstraction_text = _split_thinking_and_abstraction(final_out)

    logging.info(
        "[Traj][Final] qid=%s mode=%s thinking_len=%d abstraction_len=%d",
        question_id,
        mode,
        len(thinking_text),
        len(abstraction_text),
    )

    if mode == "unformat":
        result: Dict[str, object] = {
            "question": question_text,
            "reflection": thinking_text,
            "abstraction": abstraction_text,
        }
        if caption:
            result["captions_step"] = {"segment_1": caption}
        else:
            result["captions_step"] = {}
        return result

    return {
        "question": question_text,
        "reflection": thinking_text,
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
    parser.add_argument(
        "--exp_mode",
        type=str,
        choices=["lessons", "unformat"],
        default="lessons",
        help="Experience mode: 'lessons' keeps reflection/lessons format; 'unformat' outputs trajectory abstraction style.",
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
            exp_mode=args.exp_mode,
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
                exp_mode=args.exp_mode,
            )
            if obj is None:
                results[qid] = _empty_result_for_mode(args.exp_mode)
            else:
                results[qid] = obj
            # incremental write after each question
            _write_incremental_json(args.out, qid, results[qid])
        except Exception as e:
            logging.warning(f"abstraction generation failed for {qid}: {e}")
            results[qid] = _empty_result_for_mode(args.exp_mode)
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