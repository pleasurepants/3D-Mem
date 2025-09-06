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
        # aHash 索引缓存
        self._index = None
        self._index_path = None

    # ---------- index helpers ----------
    def _ensure_index_loaded(self):
        if isinstance(getattr(self, "_index", None), dict) and self._index is not None:
            return
        try:
            base_root = getattr(self.cfg, "retrieve_root", None) or os.path.join(self.cfg.output_parent_dir, self.cfg.exp_name)
            index_path = os.path.join(base_root, ".frontier_ahash_index.json")
            self._index_path = index_path
            if not os.path.exists(index_path):
                self._index = {}
                return
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._index = data if isinstance(data, dict) else {}
        except Exception:
            self._index = {}

    @staticmethod
    def _list_to_bits(bits_list):
        try:
            arr = np.asarray(bits_list, dtype=np.uint8).reshape(-1)
            return arr
        except Exception:
            return None

        self._replay_json: Optional[Dict] = None
        self._qid2episode_from_questions: Dict[str, str] = {}
        self._cand_bits_cache: Dict[str, np.ndarray] = {}
        self._qid2episode_from_experience: Dict[str, str] = {}
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

    def _ensure_experience_index(self):
        """从 experience_output.json 建立 question_id -> episode_id 的索引"""
        if self._qid2episode_from_experience:
            return
        try:
            base_root = getattr(self.cfg, "retrieve_root", None) or os.path.join(self.cfg.output_parent_dir, self.cfg.exp_name)
            exp_path = os.path.join(base_root, "experience_output.json")
            if not os.path.exists(exp_path):
                return
            with open(exp_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for ep_id, qdict in data.items():
                    if not isinstance(qdict, dict):
                        continue
                    for qid in qdict.keys():
                        self._qid2episode_from_experience[qid] = ep_id
        except Exception as e:
            logging.info(f"[ReplaySim] build exp index failed: {e}")

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
            # 尝试从 experience_output.json 建立索引
            self._ensure_experience_index()
            return self._qid2episode_from_experience.get(question_id)

        for epi_id, qdict in self._replay_json.items():
            if question_id in qdict:
                return epi_id
        # 最后再尝试 experience_output.json
        self._ensure_experience_index()
        return self._qid2episode_from_experience.get(question_id)

    # ---------- candidate iterator ----------
    def _candidate_iter(self, episode_id: str, exclude_qid: Optional[str] = None):
        """优先使用离线索引 <root>/.frontier_ahash_index.json 遍历候选；无需读盘解码图片。"""
        base_root = getattr(self.cfg, "retrieve_root", None) or os.path.join(self.cfg.output_parent_dir, self.cfg.exp_name)
        self._ensure_index_loaded()
        self._ensure_experience_index()
        try:
            index_size = len(self._index)
        except Exception:
            index_size = -1
        logging.info(f"[ReplaySim] Using aHash index at {getattr(self, '_index_path', '<unknown>')} (entries={index_size})")

        if not isinstance(self._index, dict) or len(self._index) == 0:
            logging.info("[ReplaySim] aHash index is empty; please run build_frontier_ahash_index.py first.")
            return

        for path, rec in self._index.items():
            try:
                qid = rec.get("question_id")
                if exclude_qid and qid == exclude_qid:
                    continue
                epi_of_q = self._qid2episode_from_experience.get(qid)
                if episode_id and epi_of_q and epi_of_q != episode_id:
                    continue
                fn = rec.get("filename")
                step_key = rec.get("step_key")
                level = rec.get("level")
                # path 现在是相对键（qid/frontier/xxx.png），统一生成相对与绝对路径
                rel_key = path
                abs_path = os.path.join(base_root, rel_key)
                yield abs_path, {
                    "episode_id": epi_of_q or episode_id,
                    "question_id": qid,
                    "step_key": step_key,
                    "level": level,
                    "filename_rel": os.path.join("frontier", fn) if fn else None,
                }
            except Exception:
                continue

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

    def _all_index_records(self):
        # 返回 (abs_path, meta, bits) 三元组迭代器
        self._ensure_index_loaded()
        for p, rec in self._index.items():
            bits_list = rec.get("bits")
            if not isinstance(bits_list, list):
                continue
            bits = self._list_to_bits(bits_list)
            if bits is None:
                continue
            yield p, rec, bits

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
        # 以离线索引为候选池
        self._ensure_index_loaded()
        if not isinstance(self._index, dict) or len(self._index) == 0:
            return []

        # build target descriptors once
        target_descs = []
        for b64 in target_b64_list:
            bits = self._image_bits_ahash_from_b64(b64)
            if bits is not None:
                target_descs.append(bits)
        if not target_descs:
            return []

        # 遍历索引项并计算 similarity
        candidates = []
        for abs_path, rec, bits in self._all_index_records():
            qid = rec.get("question_id")
            if exclude_question_id and qid == exclude_question_id:
                continue
            epi_of_q = self._qid2episode_from_experience.get(qid)
            if episode_id and epi_of_q and epi_of_q != episode_id:
                continue
            try:
                sim = float(max(self._similarity_from_bits(bits, t) for t in target_descs))
            except Exception:
                continue
            meta = {
                "episode_id": epi_of_q or episode_id,
                "question_id": qid,
                "step_key": rec.get("step_key"),
                "level": rec.get("level"),
                "filename_rel": os.path.join("frontier", rec.get("filename", "")),
            }
            candidates.append({"similarity": sim, "candidate_abs": abs_path, **meta})

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
        strategy: str = "sim",  # "sim" 或 "random"
        top_k: int = 1
    ) -> List[Dict]:
        """
        根据策略选择匹配方法
        strategy: "sim" - 选择相似度最高的前 top_k 个
                 "random" - 随机选择 top_k 个
        top_k: 返回的候选数量
        """
        valid_strategies = {"sim", "random"}
        if strategy not in valid_strategies:
            logging.warning(f"[ReplaySim] Invalid strategy '{strategy}', fallback to 'sim'. Valid: {valid_strategies}")
            strategy = "sim"

        logging.info(f"[ReplaySim] Using strategy={strategy}, top_k={top_k}")

        if strategy == "random":
            return self.search_random_match(target_b64_list, episode_id, exclude_question_id, top_k)
        else:  # strategy == "sim"
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



# v0
# def generate_step_replay_prompt(
#     best: dict,
#     cfg,
#     replay_json_path: str,
#     current_layer: str,
#     current_idx: int,
#     current_frontier_b64: str,
#     compact: bool = False,
#     ):
#     """
#     Produce ONE reusable paragraph (6–10 sentences) that encodes the two-stage choice structure
#     in a past replay AND ends with a transferable observation segment (2–3 sentences).

#     Narrative must cover:
#       - Stage 1: there were N initial directions (describe each by distinctive visual features), and the agent chose ONE.
#       - Stage 2: under that chosen direction there were M closer views (describe each by visual features), and the agent focused on ONE.
#       - TRANSFER segment (MANDATORY, 2–3 sentences): generalize why those choices were informative in visual terms and how such cues
#         can guide future choices even when the question differs. If and only if EpisodeOutcome == PASS, also state (within the same
#         segment) that these choices set up the later successful answer. Never claim the answer was completed at that step.

#     Rules:
#       - Pure text only; rely on attached visuals and factual records; no speculation.
#       - Do NOT mention or invent any image titles/captions, indices, or words like 'image/photo/picture/frontier X'.
#       - Do NOT quote labels such as 'Chosen direction' or 'Chosen closer look'.
#       - Express the total counts (N and M) as part of the narrative, but DO NOT enumerate with indices or ordinals (no 'first/second', no '1), 2) ...').
#       - Describe the chosen items deterministically by their visual features (treat them as fixed facts; do not hedge with 'one of'/'for example').
#       - Avoid vague phrases like 'similar to earlier'; explicitly name the visual properties that transfer (e.g., doorway, threshold, outdoor light,
#         railings, sink–cabinet–countertop grouping, close planar surfaces, readable clock face).
#       - Keep past tense and smooth, compact narration.
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
#         initial_keys = list(frontier.keys())  # layer0 keys (e.g., "0-layer0-1.png")
#         initial_rels = [os.path.join("frontier", k) for k in initial_keys]

#         chosen_initial = chosen.get("layer0")  # 'frontier/...'
#         chosen_detail  = chosen.get("layer1")  # 'frontier/...'

#         # If only a layer1 match is known, infer the layer0 key it came from.
#         if not chosen_initial and matched_lvl == "layer1" and matched_rel:
#             try:
#                 base = os.path.basename(matched_rel)  # e.g., "0-layer1-1_2.png"
#                 a, b = base.split("-layer1-")
#                 initial_key_guess = f"{a}-layer0-{b.split('_')[0]}.png"
#                 guessed = f"frontier/{initial_key_guess}"
#                 if os.path.basename(guessed.replace("frontier/", "")) in initial_keys:
#                     chosen_initial = guessed
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

#         has_stage2 = n_detail > 0 and (chosen_detail in detail_rels if chosen_detail else False)

#         # ========= sys_prompt (explicit structure + longer TRANSFER segment) =========
#         stage2_line = (
#             f"• Under that direction, state that {n_detail} closer views were considered and briefly characterize them by their visual features; then say the agent focused on ONE closer view and describe it by its features.\n"
#             if has_stage2
#             else
#             "• Under that direction, describe the closer views considered in that area and say the agent focused on ONE closer view, described by its visual features.\n"
#         )

#         transfer_block = (
#             "• Conclude with a short TRANSFER segment of 2–3 sentences that generalizes why those choices were informative in visual terms "
#             "and how such cues can guide future choices even when the question is different. "
#             "Name 1–2 cue types (e.g., door thresholds and outdoor light for entrances/location; close, well-lit planar surfaces for color/material; "
#             "readable faces for text/symbols; co-occurring anchors like sink/cabinet/countertop). "
#             "If and only if EpisodeOutcome is PASS, append in the same segment that these choices set up the later successful answer; "
#             "never claim the answer was completed at that step.\n"
#         )

#         sys_prompt = (
#             "You are given a current frontier view and records from a past exploration, with visuals attached.\n"
#             "Write ONE compact paragraph of 6–10 sentences, past tense, smooth narration, that makes the two-stage decision structure explicit:\n"
#             "• Start by stating that this scene was explored earlier and explicitly name the question from that time (quote it).\n"
#             f"• State that the agent initially observed {n_initial} distinct directions in that scene and briefly characterize each direction by its visual features (no numbering, no labels, no indices).\n"
#             "• Then say the agent chose ONE of those directions and describe the chosen direction by its visual features (do not mention images or titles).\n"
#             + stage2_line +
#             transfer_block +
#             "Rules:\n"
#             "- Rely only on attached visuals and provided facts; no speculation.\n"
#             "- Do NOT mention any image titles/captions, numeric indices for items, or words like 'image/photo/picture/frontier X'.\n"
#             "- Do NOT quote labels such as 'Chosen direction' or 'Chosen closer look'.\n"
#             "- Do NOT enumerate with 'first/second' or '(1)/(2)'; instead, write compact clauses separated by commas or semicolons to characterize each option.\n"
#             "- Treat the previously selected items as FIXED facts; do not hedge with 'one of'/'for example'.\n"
#             "- Avoid vague phrases like 'similar to earlier'; explicitly name the visual properties that transfer (e.g., doorway, threshold, outdoor light, railings, sink–cabinet–countertop grouping, close planar surfaces, readable clock face).\n"
#         )

#         # ========= content (facts + neutral visuals, deterministic order) =========
#         content = []

#         # (0) current candidate under consideration (empty title prevents label echo)
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

#         # helper: attach an image with an empty title (prevents the model from echoing labels)
#         def _add_img(rel_path: str):
#             abs_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, episode_id, rel_path)
#             if os.path.exists(abs_path):
#                 with open(abs_path, "rb") as f:
#                     b64 = base64.b64encode(f.read()).decode("utf-8")
#                 content.append(("", b64))

#         # (2) deterministic grounding order:
#         #     chosen direction -> chosen closer view (if any)
#         #     when not in compact mode: also attach all L0 options and L1 options (under chosen L0)
#         if chosen_initial:
#             _add_img(chosen_initial)
#         if has_stage2 and chosen_detail:
#             _add_img(chosen_detail)

#         if not compact:
#             for rel in initial_rels:
#                 _add_img(rel)

#             for rel in detail_rels:
#                 _add_img(rel)

#         return sys_prompt, content

#     except Exception as e:
#         logging.warning(f"[ReplayCtx] generate prompt failed: {e}")
#         return None






# v1
def generate_step_replay_prompt(
    best: dict,
    cfg,
    replay_json_path: str,
    current_layer: str,
    current_idx: int,
    current_frontier_b64: str,
    compact: bool = False,
    current_step_idx: 'Optional[int]' = None,  # temporal position of current step
    ):
    """
    Experience paragraph generator (ENRICHED with self-critique / self-verification principles).
    Output: ONE paragraph (6–10 sentences), past tense, smooth narration, no headings/bullets.

    Keeps:
      - Two-stage structure (Stage 1 directions -> one chosen; Stage 2 closer views -> one focused).
      - Mandatory CRITIQUE inside the same paragraph (reasons for both choices + outcome influence + temporal nuance).
      - Strict bans: no indices/labels ('frontier X', 'Chosen direction'), no enumerated lists, no speculation beyond records.
      - Factual anchoring via counts (N, M), EpisodeOutcome, StepIndex, attached visuals.

    Enrichment:
      - Self-Calibration (silent): validate against provided facts; internally correct inconsistencies.
      - Reverse-CoT (silent): restate the original question’s demand and ensure evidence serves it; adjust internally if needed.
      - Self-Verification: in the CRITIQUE, consider one plausible alternative and justify why it was inferior under the same facts.
      - Adversarial check: name a likely failure mode and state how the final rationale mitigated it.
      - Draft-to-Refine: internally remove hedging/hallucinations before emitting the final single paragraph.
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
        step_entry = steps.get(step_key)
        if not step_entry:
            return None

        frontier = step_entry.get("frontier", {})        # { '0-layer0-1.png': ['frontier/0-layer1-1_0.png', ...], ... }
        chosen   = step_entry.get("chosen_frontier", {}) # { 'layer0': 'frontier/..', 'layer1': 'frontier/..' }

        question_text = (qinfo.get("question", "") or "").strip()
        if len(question_text) > 200:
            question_text = question_text[:200] + "..."

        final_reward = (qinfo.get("final_reward", None) or "").strip().lower()

        # --- Collect stage-1 options and chosen items ---
        initial_keys = list(frontier.keys())
        initial_rels = [os.path.join("frontier", k) for k in initial_keys]

        chosen_initial = chosen.get("layer0")
        chosen_detail  = chosen.get("layer1")

        # Infer layer0 if only a matched layer1 is known
        if not chosen_initial and matched_lvl == "layer1" and matched_rel:
            try:
                base = os.path.basename(matched_rel)
                a, b = base.split("-layer1-")
                initial_key_guess = f"{a}-layer0-{b.split('_')[0]}.png"
                guessed = f"frontier/{initial_key_guess}"
                if os.path.basename(guessed.replace("frontier/", "")) in initial_keys:
                    chosen_initial = guessed
            except Exception:
                pass

        # Stage-2 options under the chosen Stage-1 direction
        initial_key_for_details = None
        if chosen_initial and chosen_initial.startswith("frontier/"):
            initial_key_for_details = chosen_initial.split("/", 1)[1]
        detail_rels = frontier.get(initial_key_for_details, []) if initial_key_for_details else []

        # Counts
        n_initial = len(initial_rels)
        n_detail  = len(detail_rels) if detail_rels else 0
        has_stage2 = n_detail > 0 and (chosen_detail in detail_rels if chosen_detail else False)

        # ========= sys_prompt (same structure; softened temporal phrasing) =========
        stage2_line = (
            f"• Under that direction, state that {n_detail} closer views were visible and briefly characterize them by their visual features; "
            "then say the agent focused on ONE closer view and describe it by its features.\n"
            if has_stage2
            else
            "• Under that direction, describe the closer views visible in that area and say the agent focused on ONE closer view, described by its visual features.\n"
        )

        critique_block = (
            "• Conclude with a CRITIQUE segment (2–4 sentences) woven into the same paragraph: "
            "explain the rationale behind selecting that direction and that closer look (visual evidence only), "
            "and discuss how these choices plausibly influenced the later PASS/FAIL outcome. "
            "Reflect on the step’s temporal position within the episode—earlier decisions typically shape coverage and information gathering, "
            "whereas later decisions often bear more direct weight on success or failure—use this nuance without stating any fixed rule. "
            "Briefly consider one plausible alternative (another direction or closer view) and justify why it was inferior under the same facts. "
            "Name one likely failure mode (e.g., over-weighting lighting cues, ignoring occlusion) and state how the final rationale mitigated it. "
            "Do not claim the answer was completed at that step.\n"
        )

        verification_addendum = (
            "Self-check before finalizing: verify that your paragraph is consistent with the provided facts "
            f"(InitialDirectionCount={n_initial}, CloserViewCount={n_detail if n_detail>0 else 0}, EpisodeOutcome, StepIndex) and the attached visuals; "
            "if any mismatch is detected, revise internally. "
            "Also perform a brief reverse check: restate what the original question demanded and ensure the described visual evidence genuinely serves that demand; "
            "if not, internally adjust the rationale so it does.\n"
        )

        sys_prompt = (
            "Role: You are an embodied agent recalling a past exploration.\n"
            "Task: Write ONE compact paragraph (6–10 sentences), in past tense, smooth narration, with an explicit two-stage decision structure and a critical reflection.\n"
            "Structure requirements:\n"
            "• Start by stating that this scene was explored earlier and explicitly name the original question from that time (quote it).\n"
            f"• State that the agent initially observed {n_initial} distinct directions in that scene and briefly characterize each direction by its visual features (no numbering, no labels, no indices).\n"
            "• Then say the agent chose ONE of those directions and describe the chosen direction by its visual features (do not mention images or titles).\n"
            + stage2_line +
            critique_block +
            "Rules:\n"
            "- Use ONLY the provided facts and attached visuals; no speculation beyond the records.\n"
            "- Do NOT mention or invent any image titles/captions, numeric indices for items, or words like 'image/photo/picture/frontier X'.\n"
            "- Do NOT quote labels such as 'Chosen direction' or 'Chosen closer look'.\n"
            "- Do NOT enumerate with 'first/second' or '(1)/(2)'; use compact descriptive clauses.\n"
            "- Treat the selected items as FIXED facts (no hedging like 'one of', 'for example').\n"
            "- Avoid vague phrasing; name concrete visual properties (doorways, thresholds, outdoor light, railings, sink–cabinet–countertop grouping, close planar surfaces, readable clock face).\n"
            "- Keep everything in ONE paragraph.\n"
            + verification_addendum
        )

        # ========= content (facts + visuals) =========
        content = []

        # (0) current frontier visual
        if current_frontier_b64:
            content.append(("", current_frontier_b64))

        # (1) factual records
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
        if isinstance(current_step_idx, int):
            step_idx_clamped = max(0, min(49, current_step_idx))
            fact_lines.append(f"StepIndex: {step_idx_clamped}")

        content.append(("Factual records (use only these facts and visuals):",))
        content.append(("\n".join(fact_lines),))

        # (2) attach chosen visuals deterministically; then (optionally) all options
        def _add_img(rel_path: str):
            abs_path = os.path.join(cfg.output_parent_dir, cfg.exp_name, episode_id, rel_path)
            if os.path.exists(abs_path):
                with open(abs_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append(("", b64))

        if chosen_initial:
            _add_img(chosen_initial)
        if has_stage2 and chosen_detail:
            _add_img(chosen_detail)

        if not compact:
            for rel in initial_rels:
                _add_img(rel)
            for rel in detail_rels:
                _add_img(rel)

        return sys_prompt, content

    except Exception as e:
        import logging
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
    strategy: str = "sim",  # "sim" 或 "random"
    top_k: int = 1
):
    """
    新版检索：
      1) 使用 sim/random 在 replay_step_info.json 候选中检索 top_k 匹配项；
      2) 不再调用任何 VLM/生成器，而是直接读取实验根目录下的 experience_output.json，
         取对应 question_id 的 steps[step_x].experience 作为上下文；
      3) 组合为一段可直接注入到 prompt 的文本（首个命中加“Most similar: ”前缀，其余用“Additionally: ”）。
    返回: (最相似候选, 合并后的上下文字符串或 None)。
    """
    # 当 top_k <= 0 时，明确不进行任何回放回忆，直接返回 None
    if top_k <= 0:
        return None, None

    # 为了日志统一，每次检索都请求最多前5个候选用于打印；实际使用仍取前 top_k
    request_k = max(5, top_k)
    candidates = searcher.search_with_strategy(
        target_b64_list=[b64_str],
        episode_id=episode_id,
        exclude_question_id=exclude_qid,
        strategy=strategy,
        top_k=request_k
    )
    if not candidates:
        return None, None
    try:
        logging.info(f"[ReplayCtx] {layer_tag}[{idx}] candidates_found={len(candidates)} (requested={request_k}, strategy={strategy})")
    except Exception:
        pass

    # 打印前5个候选的关键信息（路径/元数据/相似度）
    try:
        log_n = min(5, len(candidates))
        logging.info(f"[ReplayCtx] {layer_tag}[{idx}] TOP{log_n} candidates:")
        for rank, cand in enumerate(candidates[:log_n]):
            logging.info(
                f"  #{rank+1}: sim={cand.get('similarity', 'n/a'):.4f} | epi={cand.get('episode_id')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')} | rel={cand.get('filename_rel')} | abs={cand.get('candidate_abs')}"
            )
    except Exception:
        pass

    # 读取 experience_output.json（优先使用 cfg.retrieve_root，其次默认输出目录）
    try:
        base_root = getattr(cfg, "retrieve_root", None)
        if not base_root:
            base_root = os.path.join(cfg.output_parent_dir, cfg.exp_name)
        exp_json_path = os.path.join(base_root, "experience_output.json")
        with open(exp_json_path, "r", encoding="utf-8") as f:
            exp_data = json.load(f)
        try:
            logging.info(f"[ReplayCtx] experience_output.json loaded: {exp_json_path} (episodes={len(exp_data) if isinstance(exp_data, dict) else 'n/a'})")
        except Exception:
            pass
    except Exception as e:
        logging.warning(f"[ReplayCtx] load experience_output.json failed: {e}")
        exp_data = {}

    # 逐个候选读取对应的 experience 文本
    selected = candidates[: max(1, top_k)]
    texts = []
    for j, cand in enumerate(selected):
        epi_id = cand.get("episode_id")
        qid = cand.get("question_id")
        step_key = cand.get("step_key")
        exp_text = None
        try:
            bucket_epi = exp_data.get(epi_id, {}) if isinstance(exp_data, dict) else {}
            if not bucket_epi:
                logging.info(f"[ReplayCtx] experience: episode_id not found: {epi_id}")
            bucket_qid = bucket_epi.get(qid, {}) if isinstance(bucket_epi, dict) else {}
            if not bucket_qid:
                logging.info(f"[ReplayCtx] experience: question_id not found under episode {epi_id}: {qid}")
            bucket_steps = bucket_qid.get("steps", {}) if isinstance(bucket_qid, dict) else {}
            if not bucket_steps:
                logging.info(f"[ReplayCtx] experience: steps empty for episode {epi_id} qid {qid}")
            bucket_step = bucket_steps.get(step_key, {}) if isinstance(bucket_steps, dict) else {}
            if not bucket_step:
                logging.info(f"[ReplayCtx] experience: step not found: {step_key} for qid {qid}")
            exp_text = bucket_step.get("experience") if isinstance(bucket_step, dict) else None
        except Exception:
            exp_text = None

        if isinstance(exp_text, str) and exp_text.strip():
            prefix = "Most similar: " if j == 0 else "Additionally: "
            texts.append(prefix + exp_text.strip())
        else:
            # 若无经验文本，跳过该候选
            continue

    combined = "\n\n".join(texts) if texts else None
    best = selected[0] if selected else None
    try:
        logging.info(f"[ReplayCtx] {layer_tag}[{idx}] experience_texts: {len(texts)} / {len(selected)}")
    except Exception:
        pass
    return best, combined

def run_layer0_recall_and_aggregate(
    step: dict,
    cfg,
    frontier_imgs_0: list,
    chosen_frontier_path: str,
    strategy: str = "sim",  # "sim" 或 "random"
    top_k: int = 1
):
    """
    Do recall only for initial directions (layer0).
    Side effects: fill step["replay_*"] for layer0, and set step["replay_layer0_aggregated_context"].
    """
    # 当 top_k 为 0 时，跳过 layer0 回忆与聚合，并显式清空聚合上下文
    if top_k <= 0:
        step["replay_layer0_aggregated_context"] = None
        logging.info("[ReplaySim] top_k=0; skip layer0 recall and env context.")
        return
    # 根据外部传入的检索根目录优先读取
    _root = getattr(cfg, "retrieve_root", None)
    if not _root:
        _root = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    replay_json_path = os.path.join(_root, 'replay_step_info.json')

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
        logging.info("[ReplaySim] Searcher unavailable; skip layer0 recall.")
        return

    episode_id, exclude_qid = _resolve_episode_id(searcher, cfg, chosen_frontier_path, step)
    if episode_id is None:
        logging.warning("[ReplaySim] Cannot resolve episode_id; skip layer0 recall.")
        step["replay_layer0_aggregated_context"] = None
        return

    # ========= 新策略：对每个候选分别取 TOP5，再跨候选合并排序，取最终 top_k =========
    request_k = max(5, top_k)
    merged_candidates = []  # 收集所有候选的 TOP5
    for i, b64 in enumerate(frontier_imgs_0):
        try:
            cands = searcher.search_with_strategy(
                target_b64_list=[b64],
                episode_id=episode_id,
                exclude_question_id=exclude_qid,
                strategy=strategy,
                top_k=request_k
            )
            # 记录 per-frontier 去重后的 TOP5（按 (qid, step_key) 去重）
            try:
                log_n = min(5, len(cands))
                logging.info(f"[ReplayCtx] layer0 per-frontier idx={i} TOP{log_n}:")
                for rank, cand in enumerate(cands[:log_n]):
                    logging.info(
                        f"  #{rank+1}: sim={cand.get('similarity', 'n/a'):.4f} | epi={cand.get('episode_id')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')} | rel={cand.get('filename_rel')} | abs={cand.get('candidate_abs')}"
                    )
            except Exception:
                pass

            # per-frontier 去重并保存
            seen_keys = set()
            per_frontier_top = []
            for cand in cands:
                qid = cand.get("question_id")
                sk = cand.get("step_key")
                if not qid or not sk:
                    continue
                key = (qid, sk)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                per_frontier_top.append(cand)
                if len(per_frontier_top) >= 5:
                    break

            # 写回 per-frontier 存储（保存 qid/step 等关键字段供后续使用）
            try:
                step["replay_match_per_frontier"]["layer0"][i] = [
                    {
                        "episode_id": x.get("episode_id"),
                        "question_id": x.get("question_id"),
                        "step_key": x.get("step_key"),
                        "similarity": x.get("similarity"),
                        "level": x.get("level"),
                        "filename_rel": x.get("filename_rel"),
                    }
                    for x in per_frontier_top
                ]
            except Exception:
                pass

            # 汇总合并候选用 per-frontier 去重结果
            for cand in per_frontier_top:
                cand = dict(cand)
                cand["source_frontier_index"] = i
                merged_candidates.append(cand)
        except Exception as e:
            logging.warning(f"[ReplayCtx] layer0 per-frontier search failed idx={i}: {e}")

    if not merged_candidates:
        logging.info("[ReplayCtx] layer0 merged_candidates empty.")
        step["replay_layer0_aggregated_context"] = None
    else:
        # 统一按 (qid, step_key) 去重，保留相似度最高的一条；再排序取最终 top_k
        best_by_key = {}
        for cand in merged_candidates:
            qid = cand.get("question_id")
            sk = cand.get("step_key")
            if not qid or not sk:
                continue
            key = (qid, sk)
            prev = best_by_key.get(key)
            if prev is None or float(cand.get("similarity", 0.0)) > float(prev.get("similarity", 0.0)):
                best_by_key[key] = cand
        merged_unique = list(best_by_key.values())
        merged_unique.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        final_selected = merged_unique[: max(1, top_k)]
        try:
            logging.info(f"[ReplayCtx] layer0 FINAL merged TOP{len(final_selected)} (k={top_k}):")
            for rank, cand in enumerate(final_selected):
                logging.info(
                    f"  #{rank+1}: sim={cand.get('similarity', 'n/a'):.4f} | src_idx={cand.get('source_frontier_index')} | epi={cand.get('episode_id')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')} | rel={cand.get('filename_rel')}"
                )
        except Exception:
            pass

        # 读取 experience_output.json 一次
        try:
            _root = getattr(cfg, "retrieve_root", None) or os.path.join(cfg.output_parent_dir, cfg.exp_name)
            _exp_path = os.path.join(_root, "experience_output.json")
            with open(_exp_path, 'r', encoding='utf-8') as f:
                _exp = json.load(f)
        except Exception as e:
            logging.warning(f"[ReplayCtx] layer0 load experience_output.json failed: {e}")
            _exp = {}

        texts = []
        for j, cand in enumerate(final_selected):
            epi_id = cand.get('episode_id')
            qid = cand.get('question_id')
            step_key = cand.get('step_key')
            try:
                exp_text = (
                    _exp.get(epi_id, {})
                        .get(qid, {})
                        .get('steps', {})
                        .get(step_key, {})
                        .get('experience')
                )
            except Exception:
                exp_text = None
            if isinstance(exp_text, str) and exp_text.strip():
                prefix = "Most similar: " if j == 0 else "Additionally: "
                texts.append(prefix + exp_text.strip())

        step["replay_layer0_aggregated_context"] = "\n\n".join(texts) if texts else None
        logging.info(f"[ReplayCtx] layer0 aggregated context ready: {bool(step['replay_layer0_aggregated_context'])}")

    # -------- Global search：总是执行全局一次性检索，用于观测/兜底；仅在逐候选为空时采用 --------
    try:
        logging.info("[ReplayCtx] layer0 run global search across all layer0 candidates (always).")
        request_k = max(5, top_k)
        global_cands = searcher.search_with_strategy(
            target_b64_list=list(frontier_imgs_0),
            episode_id=episode_id,
            exclude_question_id=exclude_qid,
            strategy=strategy,
            top_k=request_k
        )
        if global_cands:
            try:
                log_n = min(5, len(global_cands))
                logging.info(f"[ReplayCtx] layer0 GLOBAL TOP{log_n}:")
                for rank, cand in enumerate(global_cands[:log_n]):
                    logging.info(
                        f"  #{rank+1}: sim={cand.get('similarity', 'n/a'):.4f} | epi={cand.get('episode_id')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')} | rel={cand.get('filename_rel')} | abs={cand.get('candidate_abs')}"
                    )
            except Exception:
                pass

            # 读取 experience_output.json
            try:
                _root = getattr(cfg, "retrieve_root", None) or os.path.join(cfg.output_parent_dir, cfg.exp_name)
                _exp_path = os.path.join(_root, "experience_output.json")
                with open(_exp_path, 'r', encoding='utf-8') as f:
                    _exp = json.load(f)
            except Exception as e:
                logging.warning(f"[ReplayCtx] global load experience_output.json failed: {e}")
                _exp = {}

            # 全局候选也进行去重后截取 top_k
            best_by_key = {}
            for cand in global_cands:
                qid = cand.get('question_id')
                sk = cand.get('step_key')
                if not qid or not sk:
                    continue
                key = (qid, sk)
                prev = best_by_key.get(key)
                if prev is None or float(cand.get('similarity', 0.0)) > float(prev.get('similarity', 0.0)):
                    best_by_key[key] = cand
            global_unique = list(best_by_key.values())
            global_unique.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)

            texts = []
            usable = 0
            for j, cand in enumerate(global_unique[:max(1, top_k)]):
                epi_id = cand.get('episode_id')
                qid = cand.get('question_id')
                step_key = cand.get('step_key')
                exp_text = None
                try:
                    exp_text = (
                        _exp.get(epi_id, {})
                            .get(qid, {})
                            .get('steps', {})
                            .get(step_key, {})
                            .get('experience')
                    )
                except Exception:
                    exp_text = None
                if isinstance(exp_text, str) and exp_text.strip():
                    prefix = "Most similar: " if j == 0 else "Additionally: "
                    texts.append(prefix + exp_text.strip())
                    usable += 1

            # 仅当逐候选聚合为空时采用全局结果
            if not step["replay_layer0_aggregated_context"] and texts:
                step["replay_layer0_aggregated_context"] = "\n\n".join(texts)
                logging.info(f"[ReplayCtx] layer0 aggregated (global-used) prepared with {usable} entries.")
        else:
            logging.info("[ReplayCtx] layer0 global search returns no candidates.")
    except Exception as e:
        logging.warning(f"[ReplayCtx] layer0 global search failed: {e}")

def run_layer1_recall_and_aggregate_for_subgroup(
    step: dict,
    cfg,
    frontier_imgs_1: list,
    layer1_indices: list,
    chosen_frontier_path: str,
    strategy: str = "sim",  # "sim" 或 "random"
    top_k: int = 1
) -> Optional[str]:

    """
    After idx0 is chosen, handle only the closer-looks subgroup (layer1 subset).
    Side effects: fill step["replay_*"]["layer1"] for the global indices used.
    Return aggregated context text for this subgroup.
    """
    # 当 top_k 为 0 时，跳过 layer1 子集回忆与聚合
    if top_k <= 0:
        logging.info("[ReplaySim] top_k=0; skip layer1 subgroup recall and env context.")
        return None
    _root = getattr(cfg, "retrieve_root", None)
    if not _root:
        _root = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    replay_json_path = os.path.join(_root, 'replay_step_info.json')

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
        logging.info("[ReplaySim] Searcher unavailable; skip layer1 subgroup recall.")
        return None

    episode_id, exclude_qid = _resolve_episode_id(searcher, cfg, chosen_frontier_path, step)
    if episode_id is None:
        logging.warning("[ReplaySim] Cannot resolve episode_id; skip layer1 subgroup recall.")
        return None

    # 新策略：对子集每个候选取 TOP5，再跨候选合并排序取最终 top_k
    request_k = max(5, top_k)
    merged_candidates = []
    for gidx in layer1_indices:
        b64 = frontier_imgs_1[gidx]
        try:
            cands = searcher.search_with_strategy(
                target_b64_list=[b64],
                episode_id=episode_id,
                exclude_question_id=exclude_qid,
                strategy=strategy,
                top_k=request_k
            )
            try:
                log_n = min(5, len(cands))
                logging.info(f"[ReplayCtx] layer1 per-frontier idx={gidx} TOP{log_n}:")
                for rank, cand in enumerate(cands[:log_n]):
                    logging.info(
                        f"  #{rank+1}: sim={cand.get('similarity', 'n/a'):.4f} | epi={cand.get('episode_id')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')} | rel={cand.get('filename_rel')} | abs={cand.get('candidate_abs')}"
                    )
            except Exception:
                pass

            # per-frontier 去重后的 TOP5
            seen_keys = set()
            per_frontier_top = []
            for cand in cands:
                qid = cand.get("question_id")
                sk = cand.get("step_key")
                if not qid or not sk:
                    continue
                key = (qid, sk)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                per_frontier_top.append(cand)
                if len(per_frontier_top) >= 5:
                    break

            # 写回 per-frontier layer1 存储
            try:
                step["replay_match_per_frontier"]["layer1"][gidx] = [
                    {
                        "episode_id": x.get("episode_id"),
                        "question_id": x.get("question_id"),
                        "step_key": x.get("step_key"),
                        "similarity": x.get("similarity"),
                        "level": x.get("level"),
                        "filename_rel": x.get("filename_rel"),
                    }
                    for x in per_frontier_top
                ]
            except Exception:
                pass

            for cand in per_frontier_top:
                cand = dict(cand)
                cand["source_frontier_index"] = gidx
                merged_candidates.append(cand)
        except Exception as e:
            logging.warning(f"[ReplayCtx] layer1 per-frontier search failed idx={gidx}: {e}")

    if not merged_candidates:
        logging.info("[ReplayCtx] layer1 merged_candidates empty.")
        return None

    # 按 (qid, step_key) 去重并取最终 top_k
    best_by_key = {}
    for cand in merged_candidates:
        qid = cand.get('question_id')
        sk = cand.get('step_key')
        if not qid or not sk:
            continue
        key = (qid, sk)
        prev = best_by_key.get(key)
        if prev is None or float(cand.get('similarity', 0.0)) > float(prev.get('similarity', 0.0)):
            best_by_key[key] = cand
    merged_unique = list(best_by_key.values())
    merged_unique.sort(key=lambda x: x.get('similarity', 0.0), reverse=True)
    final_selected = merged_unique[: max(1, top_k)]
    try:
        logging.info(f"[ReplayCtx] layer1 FINAL merged TOP{len(final_selected)} (k={top_k}):")
        for rank, cand in enumerate(final_selected):
            logging.info(
                f"  #{rank+1}: sim={cand.get('similarity', 'n/a'):.4f} | src_idx={cand.get('source_frontier_index')} | epi={cand.get('episode_id')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')} | rel={cand.get('filename_rel')}"
            )
    except Exception:
        pass

    # 读取 experience_output.json 一次
    try:
        _root = getattr(cfg, "retrieve_root", None) or os.path.join(cfg.output_parent_dir, cfg.exp_name)
        _exp_path = os.path.join(_root, "experience_output.json")
        with open(_exp_path, 'r', encoding='utf-8') as f:
            _exp = json.load(f)
    except Exception as e:
        logging.warning(f"[ReplayCtx] layer1 load experience_output.json failed: {e}")
        _exp = {}

    texts = []
    for j, cand in enumerate(final_selected):
        epi_id = cand.get('episode_id')
        qid = cand.get('question_id')
        step_key = cand.get('step_key')
        try:
            exp_text = (
                _exp.get(epi_id, {})
                    .get(qid, {})
                    .get('steps', {})
                    .get(step_key, {})
                    .get('experience')
            )
        except Exception:
            exp_text = None
        if isinstance(exp_text, str) and exp_text.strip():
            prefix = "Most similar: " if j == 0 else "Additionally: "
            texts.append(prefix + exp_text.strip())

    return "\n\n".join(texts) if texts else None

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
