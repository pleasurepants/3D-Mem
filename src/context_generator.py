# === Similarity Search over replay_step_info.json (same episode, other question_ids) ===
import os
import io
import json
import base64
import logging
import time
import random


from typing import List, Dict, Optional
from PIL import Image
import numpy as np
import openai
from openai import OpenAI
from src.const import *
client = OpenAI(
    base_url=END_POINT,
    api_key=OPENAI_KEY,
)

class FrontierSimilaritySearcher:
    """
    Search the most similar frontier image within the SAME episode_history,
    excluding the current question_id.
    """

    def __init__(self, cfg, replay_json_path: str, method: str = "ahash", feature_fn=None):
        self.cfg = cfg
        self.replay_json_path = replay_json_path
        self.method = method
        self.feature_fn = feature_fn

        self._replay_json: Optional[Dict] = None
        self._qid2episode_from_questions: Dict[str, str] = {}
        self._cand_bits_cache: Dict[str, np.ndarray] = {}
        self._replay_missing = False  # 标记 replay_json 是否缺失

    # ---------- load helpers ----------
    def _ensure_questions_index(self):
        """Build question_id -> episode_history index from cfg.questions_list_path."""
        if self._qid2episode_from_questions:
            return
        qlist_path = getattr(self.cfg, "questions_list_path", None)
        if not qlist_path or not os.path.exists(qlist_path):
            logging.warning(f"[ReplaySim] questions_list_path not found: {qlist_path}")
            return
        try:
            with open(qlist_path, "r", encoding="utf-8") as f:
                items = json.load(f)
            for it in items:
                qid = it.get("question_id")
                epi = it.get("episode_history")
                if qid and epi:
                    self._qid2episode_from_questions[qid] = epi
        except Exception as e:
            logging.warning(f"[ReplaySim] load questions_list_path failed: {e}")

    def _ensure_replay_loaded(self):
        """安全加载 replay_step_info.json，不存在则置空并标记缺失"""
        if self._replay_json is None:
            if not self.replay_json_path or not os.path.exists(self.replay_json_path):
                logging.info(f"[ReplaySim] replay json not found yet: {self.replay_json_path}")
                self._replay_json = {}
                self._replay_missing = True
                return
            with open(self.replay_json_path, "r", encoding="utf-8") as f:
                self._replay_json = json.load(f)
            self._replay_missing = False

    # ---------- episode resolve ----------
    def resolve_episode_id(self, question_id: str) -> Optional[str]:
        """优先从 questions_list_path 查找，找不到则从 replay_json 扫描"""
        if not question_id:
            return None

        self._ensure_questions_index()
        epi = self._qid2episode_from_questions.get(question_id)
        if epi:
            return epi

        self._ensure_replay_loaded()
        if self._replay_missing:
            return None

        for epi_id, qdict in self._replay_json.items():
            if question_id in qdict:
                return epi_id
        return None

    # ---------- candidate iterator ----------
    def _candidate_iter(self, episode_id: str, exclude_qid: Optional[str] = None):
        self._ensure_replay_loaded()
        if self._replay_missing or not self._replay_json or episode_id not in self._replay_json:
            return

        base_root = os.path.join(self.cfg.output_parent_dir, self.cfg.exp_name)
        for qid, qinfo in self._replay_json[episode_id].items():
            if exclude_qid and qid == exclude_qid:
                continue
            steps = qinfo.get("steps", {})
            for step_key, step_info in steps.items():
                frontier = step_info.get("frontier", {})
                # layer0 keys
                for layer0_key, layer1_list in frontier.items():
                    layer0_rel = os.path.join("frontier", layer0_key)
                    layer0_abs = os.path.join(base_root, qid, layer0_rel)
                    yield layer0_abs, {
                        "episode_id": episode_id,
                        "question_id": qid,
                        "step_key": step_key,
                        "level": "layer0",
                        "filename_rel": layer0_rel,
                    }
                    # layer1 values
                    for l1_rel in layer1_list:
                        l1_abs = os.path.join(base_root, qid, l1_rel)
                        yield l1_abs, {
                            "episode_id": episode_id,
                            "question_id": qid,
                            "step_key": step_key,
                            "level": "layer1",
                            "filename_rel": l1_rel,
                        }

    # ---------- hashing utils ----------
    @staticmethod
    def _pil_from_base64(b64: str) -> Image.Image:
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")

    @staticmethod
    def _ahash(img: Image.Image, hash_size: int = 8) -> np.ndarray:
        small = img.resize((hash_size, hash_size), Image.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)
        return (arr > arr.mean()).astype(np.uint8).reshape(-1)

    @staticmethod
    def _hamming(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.count_nonzero(a ^ b))

    @staticmethod
    def _similarity_from_bits(a: np.ndarray, b: np.ndarray) -> float:
        return 1.0 - FrontierSimilaritySearcher._hamming(a, b) / float(a.size)

    def _image_bits_ahash_from_path(self, path: str) -> Optional[np.ndarray]:
        if path in self._cand_bits_cache:
            return self._cand_bits_cache[path]
        if not os.path.exists(path):
            return None
        try:
            img = Image.open(path).convert("L")
            bits = self._ahash(img)
            self._cand_bits_cache[path] = bits
            return bits
        except Exception:
            return None

    def _image_bits_ahash_from_b64(self, b64: str) -> Optional[np.ndarray]:
        try:
            return self._ahash(self._pil_from_base64(b64))
        except Exception:
            return None

    # ---------- main search ----------
    def search_best_match(
        self,
        target_b64_list: List[str],
        episode_id: str,
        exclude_question_id: Optional[str] = None,
        top_k: int = 1,
    ) -> List[Dict]:
        self._ensure_replay_loaded()
        if self._replay_missing:
            return []

        # build target descriptors once
        target_descs = []
        for b64 in target_b64_list:
            bits = self._image_bits_ahash_from_b64(b64)
            if bits is not None:
                target_descs.append(bits)
        if not target_descs:
            return []

        candidates = []
        for cand_abs, meta in self._candidate_iter(episode_id, exclude_qid=exclude_question_id):
            cand_bits = self._image_bits_ahash_from_path(cand_abs)
            if cand_bits is None:
                continue
            sim = float(max(self._similarity_from_bits(cand_bits, t) for t in target_descs))
            candidates.append({"similarity": sim, "candidate_abs": cand_abs, **meta})
        
        # 按相似度排序并返回前 top_k 个
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]

    def search_random_match(
        self,
        target_b64_list: List[str],
        episode_id: str,
        exclude_question_id: Optional[str] = None,
        top_k: int = 1,
    ) -> List[Dict]:
        """
        随机选择 top_k 个候选，而不是基于相似度选择最佳匹配
        """
        self._ensure_replay_loaded()
        if self._replay_missing:
            return []

        # 收集所有候选
        candidates = []
        for cand_abs, meta in self._candidate_iter(episode_id, exclude_qid=exclude_question_id):
            candidates.append((cand_abs, meta))
        
        if not candidates:
            return []
        
        # 随机选择 top_k 个候选（如果候选数量少于 top_k，则返回所有候选）
        k = min(top_k, len(candidates))
        chosen_candidates = random.sample(candidates, k)
        
        # 计算相似度（用于记录，但不用于选择）
        target_descs = []
        for b64 in target_b64_list:
            bits = self._image_bits_ahash_from_b64(b64)
            if bits is not None:
                target_descs.append(bits)
        
        results = []
        for chosen_cand_abs, chosen_meta in chosen_candidates:
            sim = 0.0
            if target_descs:
                cand_bits = self._image_bits_ahash_from_path(chosen_cand_abs)
                if cand_bits is not None:
                    sim = float(max(self._similarity_from_bits(cand_bits, t) for t in target_descs))
            results.append({"similarity": sim, "candidate_abs": chosen_cand_abs, **chosen_meta})
        
        return results

    def search_with_strategy(
        self,
        target_b64_list: List[str],
        episode_id: str,
        exclude_question_id: Optional[str] = None,
        strategy: str = "best",  # "best" 或 "random"
        top_k: int = 1
    ) -> List[Dict]:
        """
        根据策略选择匹配方法
        strategy: "best" - 选择相似度最高的前 top_k 个
                 "random" - 随机选择 top_k 个
        top_k: 返回的候选数量
        """
        if strategy == "random":
            return self.search_random_match(target_b64_list, episode_id, exclude_question_id, top_k)
        else:  # default to "best"
            return self.search_best_match(target_b64_list, episode_id, exclude_question_id, top_k)

    # ---------- utils ----------
    @staticmethod
    def parse_question_id_from_path(chosen_frontier_path: str) -> Optional[str]:
        if not chosen_frontier_path:
            return None
        try:
            return os.path.basename(os.path.dirname(chosen_frontier_path))
        except Exception:
            return None

    def search_with_chosen_path(
        self,
        target_b64_list: List[str],
        chosen_frontier_path: str,
        top_k: int = 1,
    ) -> List[Dict]:
        current_qid = self.parse_question_id_from_path(chosen_frontier_path)
        if not current_qid:
            logging.warning("[ReplaySim] Cannot parse question_id from chosen_frontier_path.")
            return []

        episode_id = self.resolve_episode_id(current_qid)
        if not episode_id:
            logging.warning(f"[ReplaySim] Cannot resolve episode for question_id={current_qid}.")
            return []

        return self.search_best_match(
            target_b64_list=target_b64_list,
            episode_id=episode_id,
            exclude_question_id=current_qid,
            top_k=top_k
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
                model="qwen",  # gpt-4o-internvl-glm-qwen
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


# v2
# def generate_step_replay_prompt(
#     best: dict,
#     cfg,
#     replay_json_path: str,
#     current_layer: str,
#     current_idx: int,
#     current_frontier_b64: str,
# ):
#     """
#     Produce ONE reusable paragraph (5–7 sentences) that explicitly encodes the two-stage choice structure in the past replay:
#       - Stage 1: there were N initial directions (describe each by distinctive visual features), and the agent chose ONE of them.
#       - Stage 2: under that chosen direction there were M closer views (describe each by visual features), and the agent focused on ONE.
#       - If and only if EpisodeOutcome == PASS, end with a sentence that these choices set up the later successful answer (never claim it was answered here).

#     Rules for the paragraph:
#       - Pure text only; rely on attached visuals and factual records; no speculation.
#       - Do NOT mention or invent any image titles/captions, indices, or words like 'image/photo/picture/frontier X'.
#       - Do NOT quote labels such as 'Chosen direction' or 'Chosen closer look'.
#       - Express the total counts (N and M) as part of the narrative, but DO NOT enumerate with indices or ordinals (no 'first/second', no '1), 2) ...').
#       - Describe the chosen items deterministically by their visual features (treat them as fixed facts; do not hedge with 'one of'/'for example').
#     """
#     try:
#         import os, json, base64, logging

#         if not best:
#             return None
#         if not os.path.exists(replay_json_path):
#             logging.info(f"[ReplayCtx] replay_json not found: {replay_json_path}")
#             return None

#         with open(replay_json_path, "r", encoding="utf-8") as f:
#             replay_data = json.load(f)

#         episode_id = best.get("episode_id")
#         question_id = best.get("question_id")
#         step_key    = best.get("step_key")
#         matched_rel = best.get("filename_rel")
#         matched_lvl = best.get("level")
#         if not (episode_id and question_id and step_key and matched_rel and matched_lvl):
#             logging.info("[ReplayCtx] match lacks keys.")
#             return None

#         epi = replay_data.get(episode_id, {})
#         qinfo = epi.get(question_id)
#         if not qinfo:
#             return None

#         steps = qinfo.get("steps", {})
#         step_info = steps.get(step_key)
#         if not step_info:
#             return None

#         frontier = step_info.get("frontier", {})          # { '0-layer0-1.png': ['frontier/0-layer1-1_0.png', ...], ... }
#         chosen   = step_info.get("chosen_frontier", {})   # { 'layer0': 'frontier/..', 'layer1': 'frontier/..' }

#         question_text = (qinfo.get("question", "") or "").strip()
#         if len(question_text) > 200:
#             question_text = question_text[:200] + "..."

#         final_reward  = (qinfo.get("final_reward", None) or "").strip().lower()

#         # --- collect keys / chosen items ---
#         initial_keys = list(frontier.keys())  # layer0 keys
#         initial_rels = [os.path.join("frontier", k) for k in initial_keys]

#         chosen_initial = chosen.get("layer0")  # 'frontier/...'
#         chosen_detail  = chosen.get("layer1")  # 'frontier/...'

#         if not chosen_initial and matched_lvl == "layer1":
#             # infer layer0 key from a layer1 filename (e.g., 0-layer1-1_2.png -> 0-layer0-1.png)
#             try:
#                 base = os.path.basename(matched_rel)
#                 a, b = base.split("-layer1-")
#                 initial_key_guess = f"{a}-layer0-{b.split('_')[0]}.png"
#                 chosen_initial = f"frontier/{initial_key_guess}"
#             except Exception:
#                 pass

#         # details only for the chosen direction
#         initial_key_for_details = None
#         if chosen_initial and chosen_initial.startswith("frontier/"):
#             initial_key_for_details = chosen_initial.split("/", 1)[1]
#         detail_rels = frontier.get(initial_key_for_details, []) if initial_key_for_details else []

#         # counts for stage-1 and stage-2 (used as factual records, NOT indices)
#         n_initial = len(initial_rels)
#         n_detail  = len(detail_rels) if detail_rels else 0

#         # ========= sys_prompt (make counts explicit; forbid titles/indices) =========
#         has_stage2 = n_detail > 0 and (chosen_detail in detail_rels if chosen_detail else False)

#         stage2_line = (
#             f"• Under that direction, state that {n_detail} closer views were considered and briefly characterize them by their visual features; then say the agent focused on ONE closer view and describe it by its features.\n"
#             if has_stage2
#             else
#             "• Under that direction, describe the closer views considered in that area and say the agent focused on ONE closer view, described by its visual features.\n"
#         )

#         sys_prompt = (
#             "You are given a current frontier view and records from a past exploration, with visuals attached.\n"
#             "Write EXACTLY ONE compact paragraph of 5–7 sentences, past tense, smooth narration, that makes the two-stage decision structure explicit:\n"
#             "• Start by stating that this scene was explored earlier and explicitly name the question from that time (quote it).\n"
#             f"• State that the agent initially observed {n_initial} distinct directions in that scene and briefly characterize each direction by its visual features (no numbering, no labels, no indices).\n"
#             "• Then say the agent chose ONE of those directions and describe the chosen direction by its visual features (do not mention images or titles).\n"
#             + stage2_line +
#             "• If and only if the EpisodeOutcome is PASS, end by stating that these choices positioned the agent for a successful answer later in the episode. Do NOT claim the question was answered at this step.\n"
#             "Rules:\n"
#             "- Rely only on attached visuals and provided facts; no speculation.\n"
#             "- Do NOT mention any image titles/captions, numeric indices for items, or words like 'image/photo/picture/frontier X'.\n"
#             "- Do NOT quote labels such as 'Chosen direction' or 'Chosen closer look'.\n"
#             "- Do NOT enumerate with 'first/second' or '(1)/(2)'; instead, write compact clauses separated by commas or semicolons to characterize each option.\n"
#             "- Treat the previously selected items as FIXED facts; do not hedge with 'one of'/'for example'.\n"
#         )


#         # ========= content (facts + neutral visuals, deterministic order) =========
#         content = []

#         # (0) current candidate (empty title to avoid echo)
#         if current_frontier_b64:
#             content.append(("", current_frontier_b64))

#         # (1) factual records, including counts (to prevent hallucinated numbers)
#         fact_lines = []
#         fact_lines.append(f"Episode: {episode_id}")
#         if question_text:
#             fact_lines.append(f"Question: {question_text}")
#         if final_reward in {"pass", "fail"}:
#             outcome = "PASS" if final_reward == "pass" else "FAIL"
#             fact_lines.append(f"EpisodeOutcome: {outcome}")
#         fact_lines.append(f"InitialDirectionCount: {n_initial}")
#         if n_detail > 0:
#             fact_lines.append(f"CloserViewCount: {n_detail}")

#         content.append(("Factual records (use only these facts and visuals):",))
#         content.append(("\n".join(fact_lines),))

#         # helper: attach an image with an empty title (prevents label echo)
#         def _add_img(rel_path: str):
#             abs_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, episode_id, rel_path)
#             if os.path.exists(abs_path):
#                 with open(abs_path, "rb") as f:
#                     b64 = base64.b64encode(f.read()).decode("utf-8")
#                 content.append(("", b64))

#         # (2) REQUIRED deterministic grounding (not to be mentioned in text):
#         #     immediately after factual records -> chosen direction -> chosen closer view (if any)
#         if chosen_initial:
#             _add_img(chosen_initial)
#         if has_stage2 and chosen_detail:
#             _add_img(chosen_detail)

#         # (3) all first-stage options (neutral)
#         for rel in initial_rels:
#             _add_img(rel)

#         # (4) all second-stage options under the chosen direction (neutral)
#         for rel in detail_rels:
#             _add_img(rel)

#         return sys_prompt, content

#     except Exception as e:
#         logging.warning(f"[ReplayCtx] generate prompt failed: {e}")
#         return None


# v3
def generate_step_replay_prompt(
    best: dict,
    cfg,
    replay_json_path: str,
    current_layer: str,
    current_idx: int,
    current_frontier_b64: str,
):
    """
    Produce ONE reusable paragraph (6–10 sentences) that encodes the two-stage choice structure
    in a past replay AND ends with a transferable observation segment (2–3 sentences).

    Narrative must cover:
      - Stage 1: there were N initial directions (describe each by distinctive visual features), and the agent chose ONE.
      - Stage 2: under that chosen direction there were M closer views (describe each by visual features), and the agent focused on ONE.
      - TRANSFER segment (MANDATORY, 2–3 sentences): generalize why those choices were informative in visual terms and how such cues
        can guide future choices even when the question differs. If and only if EpisodeOutcome == PASS, also state (within the same
        segment) that these choices set up the later successful answer. Never claim the answer was completed at that step.

    Rules:
      - Pure text only; rely on attached visuals and factual records; no speculation.
      - Do NOT mention or invent any image titles/captions, indices, or words like 'image/photo/picture/frontier X'.
      - Do NOT quote labels such as 'Chosen direction' or 'Chosen closer look'.
      - Express the total counts (N and M) as part of the narrative, but DO NOT enumerate with indices or ordinals (no 'first/second', no '1), 2) ...').
      - Describe the chosen items deterministically by their visual features (treat them as fixed facts; do not hedge with 'one of'/'for example').
      - Avoid vague phrases like 'similar to earlier'; explicitly name the visual properties that transfer (e.g., doorway, threshold, outdoor light,
        railings, sink–cabinet–countertop grouping, close planar surfaces, readable clock face).
      - Keep past tense and smooth, compact narration.
    """
    try:
        import os, json, base64, logging

        if not best:
            return None
        if not os.path.exists(replay_json_path):
            logging.info(f"[ReplayCtx] replay_json not found: {replay_json_path}")
            return None

        with open(replay_json_path, "r", encoding="utf-8") as f:
            replay_data = json.load(f)

        episode_id = best.get("episode_id")
        question_id = best.get("question_id")
        step_key    = best.get("step_key")
        matched_rel = best.get("filename_rel")
        matched_lvl = best.get("level")
        if not (episode_id and question_id and step_key and matched_rel and matched_lvl):
            logging.info("[ReplayCtx] match lacks keys.")
            return None

        epi = replay_data.get(episode_id, {})
        qinfo = epi.get(question_id)
        if not qinfo:
            return None

        steps = qinfo.get("steps", {})
        step_info = steps.get(step_key)
        if not step_info:
            return None

        frontier = step_info.get("frontier", {})          # { '0-layer0-1.png': ['frontier/0-layer1-1_0.png', ...], ... }
        chosen   = step_info.get("chosen_frontier", {})   # { 'layer0': 'frontier/..', 'layer1': 'frontier/..' }

        question_text = (qinfo.get("question", "") or "").strip()
        if len(question_text) > 200:
            question_text = question_text[:200] + "..."

        final_reward  = (qinfo.get("final_reward", None) or "").strip().lower()

        # --- collect keys / chosen items ---
        initial_keys = list(frontier.keys())  # layer0 keys (e.g., "0-layer0-1.png")
        initial_rels = [os.path.join("frontier", k) for k in initial_keys]

        chosen_initial = chosen.get("layer0")  # 'frontier/...'
        chosen_detail  = chosen.get("layer1")  # 'frontier/...'

        # If only a layer1 match is known, infer the layer0 key it came from.
        if not chosen_initial and matched_lvl == "layer1" and matched_rel:
            try:
                base = os.path.basename(matched_rel)  # e.g., "0-layer1-1_2.png"
                a, b = base.split("-layer1-")
                initial_key_guess = f"{a}-layer0-{b.split('_')[0]}.png"
                guessed = f"frontier/{initial_key_guess}"
                if os.path.basename(guessed.replace("frontier/", "")) in initial_keys:
                    chosen_initial = guessed
            except Exception:
                pass

        # details only for the chosen direction
        initial_key_for_details = None
        if chosen_initial and chosen_initial.startswith("frontier/"):
            initial_key_for_details = chosen_initial.split("/", 1)[1]
        detail_rels = frontier.get(initial_key_for_details, []) if initial_key_for_details else []

        # counts for stage-1 and stage-2 (used as factual records, NOT indices)
        n_initial = len(initial_rels)
        n_detail  = len(detail_rels) if detail_rels else 0

        has_stage2 = n_detail > 0 and (chosen_detail in detail_rels if chosen_detail else False)

        # ========= sys_prompt (explicit structure + longer TRANSFER segment) =========
        stage2_line = (
            f"• Under that direction, state that {n_detail} closer views were considered and briefly characterize them by their visual features; then say the agent focused on ONE closer view and describe it by its features.\n"
            if has_stage2
            else
            "• Under that direction, describe the closer views considered in that area and say the agent focused on ONE closer view, described by its visual features.\n"
        )

        transfer_block = (
            "• Conclude with a short TRANSFER segment of 2–3 sentences that generalizes why those choices were informative in visual terms "
            "and how such cues can guide future choices even when the question is different. "
            "Name 1–2 cue types (e.g., door thresholds and outdoor light for entrances/location; close, well-lit planar surfaces for color/material; "
            "readable faces for text/symbols; co-occurring anchors like sink/cabinet/countertop). "
            "If and only if EpisodeOutcome is PASS, append in the same segment that these choices set up the later successful answer; "
            "never claim the answer was completed at that step.\n"
        )

        sys_prompt = (
            "You are given a current frontier view and records from a past exploration, with visuals attached.\n"
            "Write ONE compact paragraph of 6–10 sentences, past tense, smooth narration, that makes the two-stage decision structure explicit:\n"
            "• Start by stating that this scene was explored earlier and explicitly name the question from that time (quote it).\n"
            f"• State that the agent initially observed {n_initial} distinct directions in that scene and briefly characterize each direction by its visual features (no numbering, no labels, no indices).\n"
            "• Then say the agent chose ONE of those directions and describe the chosen direction by its visual features (do not mention images or titles).\n"
            + stage2_line +
            transfer_block +
            "Rules:\n"
            "- Rely only on attached visuals and provided facts; no speculation.\n"
            "- Do NOT mention any image titles/captions, numeric indices for items, or words like 'image/photo/picture/frontier X'.\n"
            "- Do NOT quote labels such as 'Chosen direction' or 'Chosen closer look'.\n"
            "- Do NOT enumerate with 'first/second' or '(1)/(2)'; instead, write compact clauses separated by commas or semicolons to characterize each option.\n"
            "- Treat the previously selected items as FIXED facts; do not hedge with 'one of'/'for example'.\n"
            "- Avoid vague phrases like 'similar to earlier'; explicitly name the visual properties that transfer (e.g., doorway, threshold, outdoor light, railings, sink–cabinet–countertop grouping, close planar surfaces, readable clock face).\n"
        )

        # ========= content (facts + neutral visuals, deterministic order) =========
        content = []

        # (0) current candidate under consideration (empty title prevents label echo)
        if current_frontier_b64:
            content.append(("", current_frontier_b64))

        # (1) factual records, including counts (to prevent hallucinated numbers)
        fact_lines = []
        fact_lines.append(f"Episode: {episode_id}")
        if question_text:
            fact_lines.append(f"Question: {question_text}")
        if final_reward in {"pass", "fail"}:
            outcome = "PASS" if final_reward == "pass" else "FAIL"
            fact_lines.append(f"EpisodeOutcome: {outcome}")
        fact_lines.append(f"InitialDirectionCount: {n_initial}")
        if n_detail > 0:
            fact_lines.append(f"CloserViewCount: {n_detail}")

        content.append(("Factual records (use only these facts and visuals):",))
        content.append(("\n".join(fact_lines),))

        # helper: attach an image with an empty title (prevents the model from echoing labels)
        def _add_img(rel_path: str):
            abs_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, episode_id, rel_path)
            if os.path.exists(abs_path):
                with open(abs_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append(("", b64))

        # (2) deterministic grounding order:
        #     chosen direction -> chosen closer view (if any) -> all L0 options -> all L1 options (under the chosen L0)
        if chosen_initial:
            _add_img(chosen_initial)
        if has_stage2 and chosen_detail:
            _add_img(chosen_detail)

        for rel in initial_rels:
            _add_img(rel)

        for rel in detail_rels:
            _add_img(rel)

        return sys_prompt, content

    except Exception as e:
        logging.warning(f"[ReplayCtx] generate prompt failed: {e}")
        return None



# -------- Orchestrator helpers (encapsulated) --------
def _shorten(text: str, max_len: int = 400) -> str:
    if not text:
        return ""
    t = text.strip()
    return (t[:max_len] + " ...") if len(t) > max_len else t

def aggregate_recall_contexts_for_layer(
    layer_alias: str,              # 兼容旧签名
    contexts: list,                # List[Optional[str]]
    indices: Optional[list] = None,
    prefix: str = "Frontier ",
    max_len: Optional[int] = 300,  # None 或 <=0 表示不截断
    include_empty: bool = False
) -> Optional[str]:
    """
    返回按索引排列的回忆 context：
        Frontier i: <context>
    - 不添加额外标题
    - 无有效条目则返回 None
    """
    # 选出要汇总的 (index, context) 对
    if indices is None:
        pairs = list(enumerate(contexts))
    else:
        pairs = [(i, contexts[i] if 0 <= i < len(contexts) else None) for i in indices]

    lines = []
    for i, ctx in pairs:
        if ctx and isinstance(ctx, str) and ctx.strip():
            text = ctx.strip()
            if isinstance(max_len, int) and max_len > 0:
                text = _shorten(text, max_len)
            # max_len 为 None 或 <=0 时不截断
            lines.append(f"{prefix}{i}: {text}")
        elif include_empty:
            lines.append(f"{prefix}{i}: <no context>")

    if not lines:
        return None
    return "\n".join(lines)


def _build_searcher_if_ready(cfg, replay_json_path: str):
    """Return (searcher | None)."""
    if not os.path.exists(replay_json_path):
        logging.info(f"[ReplaySim] replay_json not found yet: {replay_json_path}")
        return None
    return FrontierSimilaritySearcher(cfg, replay_json_path, method="ahash")

def _resolve_episode_id(searcher, cfg, chosen_frontier_path, step):
    """Resolve episode_id with preference order: chosen_frontier_path -> step['question_id']."""
    exclude_qid = None
    episode_id = None
    if chosen_frontier_path:
        exclude_qid = FrontierSimilaritySearcher.parse_question_id_from_path(chosen_frontier_path)
        episode_id = searcher.resolve_episode_id(exclude_qid) if exclude_qid else None
    if episode_id is None:
        exclude_qid = exclude_qid or step.get('question_id')
        if exclude_qid:
            episode_id = searcher.resolve_episode_id(exclude_qid)
    return episode_id, exclude_qid

def _process_candidate_one(
    searcher,
    episode_id: str,
    exclude_qid: str,
    b64_str: str,
    cfg,
    replay_json_path: str,
    layer_tag: str,
    idx: int,
    strategy: str = "best",  # "best" 或 "random"
    top_k: int = 1
):
    """
    For one candidate:
      1) find best match in same episode (or random match based on strategy)
      2) build narrative prompt (sys, content)
      3) call VLM -> get short recall context text
    Return (best, ctx_text | None).
    """
    candidates = searcher.search_with_strategy(
        target_b64_list=[b64_str],
        episode_id=episode_id,
        exclude_question_id=exclude_qid,
        strategy=strategy,
        top_k=top_k
    )
    if not candidates:
        return None, None

    # 使用第一个候选（最相似或随机选择的第一个）
    best = candidates[0]
    
    prompt_pack = generate_step_replay_prompt(
        best=best,
        cfg=cfg,
        replay_json_path=replay_json_path,
        current_layer=layer_tag,
        current_idx=idx,
        current_frontier_b64=b64_str,
    )
    if not prompt_pack:
        return best, None

    sys_p, cont = prompt_pack
    ctx_text = call_openai_api(sys_p, cont)
    return best, ctx_text

def run_layer0_recall_and_aggregate(
    step: dict,
    cfg,
    frontier_imgs_0: list,
    chosen_frontier_path: str,
    strategy: str = "best",  # "best" 或 "random"
    top_k: int = 1
):
    """
    Do recall only for initial directions (layer0).
    Side effects: fill step["replay_*"] for layer0, and set step["replay_layer0_aggregated_context"].
    """
    replay_json_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, 'replay_step_info.json')

    # init containers
    step["replay_match_per_frontier"] = step.get("replay_match_per_frontier", {})
    step["replay_context_prompt_per_frontier"] = step.get("replay_context_prompt_per_frontier", {})
    step["replay_context_text_per_frontier"] = step.get("replay_context_text_per_frontier", {})

    step["replay_match_per_frontier"]["layer0"] = [None] * len(frontier_imgs_0)
    step["replay_context_prompt_per_frontier"]["layer0"] = [None] * len(frontier_imgs_0)
    step["replay_context_text_per_frontier"]["layer0"] = [None] * len(frontier_imgs_0)

    searcher = _build_searcher_if_ready(cfg, replay_json_path)
    if searcher is None:
        step["replay_layer0_aggregated_context"] = None
        return

    episode_id, exclude_qid = _resolve_episode_id(searcher, cfg, chosen_frontier_path, step)
    if episode_id is None:
        logging.warning("[ReplaySim] Cannot resolve episode_id; skip layer0 recall.")
        step["replay_layer0_aggregated_context"] = None
        return

    # per candidate
    for i, b64 in enumerate(frontier_imgs_0):
        try:
            best, ctx_text = _process_candidate_one(
                searcher=searcher,
                episode_id=episode_id,
                exclude_qid=exclude_qid,
                b64_str=b64,
                cfg=cfg,
                replay_json_path=replay_json_path,
                layer_tag="layer0",
                idx=i,
                strategy=strategy,
                top_k=top_k
            )
            step["replay_match_per_frontier"]["layer0"][i] = best
            step["replay_context_prompt_per_frontier"]["layer0"][i] = {"sys": "<hidden>", "content_len": -1} if best else None
            step["replay_context_text_per_frontier"]["layer0"][i] = ctx_text
        except Exception as e:
            logging.warning(f"[ReplayCtx] layer0[{i}] failed: {e}")

    # aggregate to natural-language summary (no 'layer' words)
    step["replay_layer0_aggregated_context"] = aggregate_recall_contexts_for_layer(
        layer_alias="initial directions",
        contexts=step["replay_context_text_per_frontier"]["layer0"],
        max_len=None
    )

def run_layer1_recall_and_aggregate_for_subgroup(
    step: dict,
    cfg,
    frontier_imgs_1: list,
    layer1_indices: list,
    chosen_frontier_path: str,
    strategy: str = "best",  # "best" 或 "random"
    top_k: int = 1
) -> Optional[str]:

    """
    After idx0 is chosen, handle only the closer-looks subgroup (layer1 subset).
    Side effects: fill step["replay_*"]["layer1"] for the global indices used.
    Return aggregated context text for this subgroup.
    """
    replay_json_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, 'replay_step_info.json')

    # ensure containers
    step["replay_match_per_frontier"] = step.get("replay_match_per_frontier", {})
    step["replay_context_prompt_per_frontier"] = step.get("replay_context_prompt_per_frontier", {})
    step["replay_context_text_per_frontier"] = step.get("replay_context_text_per_frontier", {})

    # allocate full length arrays (align to global layer1 length)
    N1 = len(frontier_imgs_1)
    step["replay_match_per_frontier"]["layer1"] = step["replay_match_per_frontier"].get("layer1") or [None] * N1
    step["replay_context_prompt_per_frontier"]["layer1"] = step["replay_context_prompt_per_frontier"].get("layer1") or [None] * N1
    step["replay_context_text_per_frontier"]["layer1"] = step["replay_context_text_per_frontier"].get("layer1") or [None] * N1

    searcher = _build_searcher_if_ready(cfg, replay_json_path)
    if searcher is None:
        return None

    episode_id, exclude_qid = _resolve_episode_id(searcher, cfg, chosen_frontier_path, step)
    if episode_id is None:
        logging.warning("[ReplaySim] Cannot resolve episode_id; skip layer1 subgroup recall.")
        return None

    # per candidate in subgroup
    for gidx in layer1_indices:
        b64 = frontier_imgs_1[gidx]
        try:
            best, ctx_text = _process_candidate_one(
                searcher=searcher,
                episode_id=episode_id,
                exclude_qid=exclude_qid,
                b64_str=b64,
                cfg=cfg,
                replay_json_path=replay_json_path,
                layer_tag="layer1",
                idx=gidx,
                strategy=strategy,
                top_k=top_k
            )
            step["replay_match_per_frontier"]["layer1"][gidx] = best
            step["replay_context_prompt_per_frontier"]["layer1"][gidx] = {"sys": "<hidden>", "content_len": -1} if best else None
            step["replay_context_text_per_frontier"]["layer1"][gidx] = ctx_text
        except Exception as e:
            logging.warning(f"[ReplayCtx] layer1[{gidx}] failed: {e}")

    # aggregate subgroup natural-language summary
    layer1_texts_all = step["replay_context_text_per_frontier"]["layer1"]
    return aggregate_recall_contexts_for_layer(
        layer_alias="closer looks",
        contexts=layer1_texts_all,
        indices=layer1_indices,
        max_len=None
    )

# ========= 使用示例 =========
def example_usage():
    """
    展示如何使用新的 top_k 和 strategy 参数
    """
    # 假设你已经有了配置和搜索器
    # cfg = your_config
    # searcher = FrontierSimilaritySearcher(cfg, replay_json_path)
    
    # 示例 1: 获取前3个最相似的候选
    # best_candidates = searcher.search_best_match(
    #     target_b64_list=[your_b64_image],
    #     episode_id="episode_123",
    #     top_k=3
    # )
    
    # 示例 2: 随机选择2个候选
    # random_candidates = searcher.search_random_match(
    #     target_b64_list=[your_b64_image],
    #     episode_id="episode_123",
    #     top_k=2
    # )
    
    # 示例 3: 使用策略选择前5个候选
    # candidates = searcher.search_with_strategy(
    #     target_b64_list=[your_b64_image],
    #     episode_id="episode_123",
    #     strategy="best",  # 或 "random"
    #     top_k=5
    # )
    
    # 示例 4: 在 layer0 回忆中使用新参数
    # run_layer0_recall_and_aggregate(
    #     step=your_step,
    #     cfg=your_cfg,
    #     frontier_imgs_0=your_images,
    #     chosen_frontier_path=your_path,
    #     strategy="random",
    #     top_k=3
    # )
    
    pass
