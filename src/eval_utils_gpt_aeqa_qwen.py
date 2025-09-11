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
import numpy as np
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


# ================== 最简检索实现（独立于 context_generator） ==================
def _pil_bits_from_b64(b64: str, hash_size: int = 8):
    try:
        img_bytes = base64.b64decode(b64)
        im = Image.open(BytesIO(img_bytes)).convert("L").resize((hash_size, hash_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        return (arr > arr.mean()).astype(np.uint8).reshape(-1)
    except Exception:
        return None


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a ^ b))


def _similarity(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - _hamming(a, b) / float(a.size)


def _load_frontier_index(cfg):
    root = getattr(cfg, "retrieve_root", None) or os.path.join(cfg.output_parent_dir, cfg.exp_name)
    idx_path = os.path.join(root, ".frontier_ahash_index.json")
    if not os.path.exists(idx_path):
        logging.info(f"[SimpleRecall] index not found: {idx_path}")
        return {}
    try:
        with open(idx_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.warning(f"[SimpleRecall] load index failed: {e}")
        return {}


def _load_experience(cfg):
    root = getattr(cfg, "retrieve_root", None) or os.path.join(cfg.output_parent_dir, cfg.exp_name)
    exp_filename = getattr(cfg, "experience_filename", "experience_output.json")
    exp_path = os.path.join(root, exp_filename)
    if not os.path.exists(exp_path):
        logging.info(f"[SimpleRecall] experience_output.json not found: {exp_path}")
        return {}
    try:
        with open(exp_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.warning(f"[SimpleRecall] load experience failed: {e}")
        return {}


_AEQA_QID2QUESTION = None


def _load_questions_en() -> dict:
    global _AEQA_QID2QUESTION
    if isinstance(_AEQA_QID2QUESTION, dict):
        return _AEQA_QID2QUESTION
    try:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        qpath = os.path.join(repo_root, "data", "aeqa_questions-168.json")
        with open(qpath, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        qmap = {}
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                qid = it.get("question_id")
                qtext = it.get("question")
                if isinstance(qid, str) and isinstance(qtext, str) and qid and qtext:
                    qmap[qid] = qtext
        _AEQA_QID2QUESTION = qmap
        return qmap
    except Exception as e:
        logging.warning(f"[SimpleRecall] load questions file failed: {e}")
        _AEQA_QID2QUESTION = {}
        return _AEQA_QID2QUESTION


def _tokenize(text: str):
    if not isinstance(text, str):
        return []
    import re as _re
    t = _re.sub(r"[^\w\s]", " ", text.lower())
    return [w for w in t.split() if w]


def _cosine_sim_tokens(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    from collections import Counter
    ca, cb = Counter(ta), Counter(tb)
    import math
    keys = set(ca.keys()) | set(cb.keys())
    dot = sum(ca[k] * cb[k] for k in keys)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def _load_vector_store(cfg):
    """
    加载使用 build_retrieve_store 生成的向量仓库。
    期望结构：<root>/retrieve/png|question/{embeddings.npy, meta.json, encoders.json, index.faiss?}
    返回：{
        'png': {'emb': np.ndarray [N_img, D], 'meta': list[dict], 'enc': dict},
        'question': {'emb': np.ndarray [N_q, Dq], 'meta': list[dict], 'enc': dict},
        'root': <retrieve_dir>
    } 或 None
    """
    root = getattr(cfg, "retrieve_root", None) or os.path.join(cfg.output_parent_dir, cfg.exp_name)
    retrieve_dir = os.path.join(root, "retrieve")
    try:
        import numpy as _np
        import json as _json
        def _load_one(sub):
            subdir = os.path.join(retrieve_dir, sub)
            emb = _np.load(os.path.join(subdir, 'embeddings.npy'))
            with open(os.path.join(subdir, 'meta.json'), 'r', encoding='utf-8') as f:
                meta = _json.load(f)
            with open(os.path.join(subdir, 'encoders.json'), 'r', encoding='utf-8') as f:
                enc = _json.load(f)
            return {'emb': emb.astype(_np.float32), 'meta': meta, 'enc': enc}
        png = _load_one('png')
        qst = _load_one('question')
        return {'png': png, 'question': qst, 'root': retrieve_dir}
    except Exception as e:
        logging.warning(f"[VecRetrieve] load vector store failed: {e}")
        return None


def _embed_images_with_clip(pil_list, model_name: str, pretrained: str = None, device: str = 'cpu'):
    try:
        import torch
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=(pretrained or 'openai'))
        model = model.to(device)
        model.eval()
        import torch as _torch
        ims = [preprocess(img).unsqueeze(0) for img in pil_list]
        batch = _torch.cat(ims, dim=0).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feats.cpu().numpy().astype(np.float32)
    except Exception as e:
        logging.warning(f"[VecRetrieve] open_clip image embed failed: {e}")
        return None


def _embed_text_with_clip(texts, model_name: str, pretrained: str = None, tokenizer_model: str = None, device: str = 'cpu'):
    """
    尝试使用 open_clip 对文本编码；失败返回 None。
    不依赖 PIL。
    """
    try:
        import torch
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=(pretrained or 'openai'))
        model = model.to(device)
        model.eval()
        tokenizer = open_clip.get_tokenizer(tokenizer_model or model_name)
        with torch.no_grad():
            toks = tokenizer(texts).to(device)
            feats = model.encode_text(toks).float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feats.cpu().numpy().astype(np.float32)
    except Exception as e:
        logging.warning(f"[VecRetrieve] open_clip text embed failed: {e}")
        return None


def _embed_text_with_sbert(texts, model_name: str, device: str = 'cpu'):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        logging.warning(f"[VecRetrieve] SBERT text embed failed: {e}")
        return None


def _device_auto():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def _rank_by_vectors_for_question(current_question: str, store: dict, top_k: int = 5, rrf_k: int = 60):
    """
    使用问题文本向量对两路库（png 与 question）做检索，并用 RRF 融合，返回 top_k 的 question_id 列表及分数。
    若无法向量化，则返回空列表。
    """
    if not isinstance(current_question, str) or not current_question.strip():
        return []
    try:
        dev = _device_auto()
        # 先决定文本编码器
        q_enc = store['question']['enc'] or {}
        text_encoder = str(q_enc.get('text_encoder', 'clip')).lower()
        txt_vec = None
        if text_encoder == 'clip':
            q_enc = store['question']['enc'] or {}
            p_enc = store['png']['enc'] or {}
            model_name = str(q_enc.get('clip_model', p_enc.get('clip_model', 'ViT-B-32')))
            pretrained = q_enc.get('open_clip_pretrained') or p_enc.get('open_clip_pretrained') or None
            tok_model = q_enc.get('open_clip_tokenizer_model') or p_enc.get('open_clip_tokenizer_model') or 'ViT-B-32'
            txt_vec = _embed_text_with_clip(
                [current_question], model_name=model_name, pretrained=pretrained, tokenizer_model=tok_model, device=dev
            )
        else:
            sbert_name = str(q_enc.get('sbert_model', 'sentence-transformers/all-MiniLM-L6-v2'))
            txt_vec = _embed_text_with_sbert([current_question], model_name=sbert_name, device=dev)
        if txt_vec is None:
            return []
        qv = txt_vec[0]
        # 两路相似度（若文本编码器不是 CLIP，则禁用 text→png 通道）
        import numpy as _np
        def _cos_sim(v, M):
            return (M @ v.astype(_np.float32))  # M 和 v 已 L2 归一化
        text_encoder = str((store['question']['enc'] or {}).get('text_encoder', 'clip')).lower()
        sims_img = None
        if text_encoder == 'clip':
            sims_img = _cos_sim(qv, store['png']['emb'])  # text-image CLIP 相似度
        sims_txt = _cos_sim(qv, store['question']['emb'])  # text-text 相似度（CLIP或SBERT）
        # 排名到 RRF
        rank_img = _np.argsort(-sims_img) if sims_img is not None else _np.array([], dtype=int)
        rank_txt = _np.argsort(-sims_txt)
        # 取前若干用于融合
        topN_img = rank_img[: max(1000, top_k)] if sims_img is not None and rank_img.size > 0 else []
        topN_txt = rank_txt[: max(1000, top_k)]
        # 统计 question_id 层面（png 通道需要把每张图映射到其 qid）
        qid_rrf = {}
        # img 通道：qid 取最优 rank
        qid_best_rank_img = {}
        if sims_img is not None and len(topN_img) > 0:
            for ri, idx in enumerate(topN_img, start=1):
                qid = store['png']['meta'][int(idx)].get('question_id')
                if qid is None:
                    continue
                if (qid not in qid_best_rank_img) or (ri < qid_best_rank_img[qid]):
                    qid_best_rank_img[qid] = ri
        # txt 通道：qid 对应其在 question 库中的行
        qid_to_row = { m.get('question_id'): i for i, m in enumerate(store['question']['meta']) }
        qid_best_rank_txt = {}
        for rt, idx in enumerate(topN_txt, start=1):
            qid = store['question']['meta'][int(idx)].get('question_id')
            if qid is None:
                continue
            if (qid not in qid_best_rank_txt) or (rt < qid_best_rank_txt[qid]):
                qid_best_rank_txt[qid] = rt
        # 融合打分
        for qid in set(list(qid_best_rank_img.keys()) + list(qid_best_rank_txt.keys())):
            ri = qid_best_rank_img.get(qid, 10**9)
            rt = qid_best_rank_txt.get(qid, 10**9)
            score = (1.0 / (rrf_k + ri)) + (1.0 / (rrf_k + rt))
            qid_rrf[qid] = score
        # 日志：展示 image-only 和 text-only 的前 top_k（按 qid 聚合）
        try:
            if sims_img is not None and len(topN_img) > 0:
                qid_best_sim_img = {}
                for idx in topN_img:
                    meta = store['png']['meta'][int(idx)]
                    qid = meta.get('question_id')
                    if qid is None:
                        continue
                    s = float(sims_img[int(idx)])
                    if (qid not in qid_best_sim_img) or (s > qid_best_sim_img[qid]):
                        qid_best_sim_img[qid] = s
                img_only_top = sorted(qid_best_sim_img.items(), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
                logging.info(f"[VecRetrieve][RRF] image-only top {len(img_only_top)}:")
                for rank, (qid, s) in enumerate(img_only_top, 1):
                    logging.info(f"  #{rank}: qid={qid} | clip_sim={s:.4f}")
            qid_best_sim_txt = {}
            for idx in topN_txt:
                qid = store['question']['meta'][int(idx)].get('question_id')
                if qid is None:
                    continue
                s = float(sims_txt[int(idx)])
                if (qid not in qid_best_sim_txt) or (s > qid_best_sim_txt[qid]):
                    qid_best_sim_txt[qid] = s
            text_only_top = sorted(qid_best_sim_txt.items(), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
            logging.info(f"[VecRetrieve][RRF] text-only top {len(text_only_top)}:")
            for rank, (qid, s) in enumerate(text_only_top, 1):
                logging.info(f"  #{rank}: qid={qid} | qsim={s:.4f}")
        except Exception:
            pass

        # 取 top_k qid
        ranked = sorted(qid_rrf.items(), key=lambda x: x[1], reverse=True)[: max(1, top_k)]
        return ranked
    except Exception as e:
        logging.warning(f"[VecRetrieve] ranking failed: {e}")
        return []


def simple_recall_and_aggregate(frontier_imgs_b64, cfg, exclude_question_id=None, top_k=1, strategy: str = 'sim', current_question: str = None, rrf_k: int = 60):
    if not frontier_imgs_b64 or top_k <= 0:
        return None

    # 'sim' = 图像向量检索（当前BVF/CVF图片 → 训练集PNG向量）+ 文本相似（问题→问题文本）做 RRF 融合；'question-first' = 先召回top-3相似question，再在这些question的PNG中做CLIP检索；'random' 维持不变
    if (strategy or 'sim') == 'question-first':
        # 步骤1: 先用问题相似度召回top-3相似question
        store = _load_vector_store(cfg)
        if not store:
            logging.info("[VecRetrieve] vector store not available")
        else:
            try:
                # 日志：记录当前问题
                logging.info(f"[VecRetrieve][question-first] Current question: {current_question or 'N/A'}")
                ranked_qids = _rank_by_vectors_for_question(current_question=current_question or '', store=store, top_k=3, rrf_k=rrf_k)
                # 日志：记录retrieve回来的top-3相似question
                if ranked_qids:
                    logging.info(f"[VecRetrieve][question-first] Retrieved top-3 similar questions:")
                    for rank, (qid, score) in enumerate(ranked_qids, 1):
                        qtext = next((m['question'] for m in store['question']['meta'] if m['question_id'] == qid), 'N/A')
                        logging.info(f"  #{rank}: qid={qid} | qsim={score:.4f} | question={qtext}")
                if not ranked_qids or not frontier_imgs_b64:
                    logging.info("[VecRetrieve] question ranking empty or no frontiers")
                else:
                    # 步骤2: 收集这些question对应的PNG向量索引
                    qid_to_indices = {}
                    for idx, meta in enumerate(store['png']['meta']):
                        qid = meta.get('question_id')
                        if qid and qid in [q for q, _ in ranked_qids]:
                            if qid not in qid_to_indices:
                                qid_to_indices[qid] = []
                            qid_to_indices[qid].append(idx)

                    relevant_indices = []
                    for indices in qid_to_indices.values():
                        relevant_indices.extend(indices)
                    if not relevant_indices:
                        logging.info("[VecRetrieve] no relevant PNGs")
                    else:
                        corpus = store['png']['emb'][relevant_indices]
                        meta_list = [store['png']['meta'][idx] for idx in relevant_indices]

                        # 步骤3: 对当前frontier图片做CLIP相似度检索
                        dev = _device_auto()
                        from PIL import Image as _Image
                        from io import BytesIO as _BytesIO
                        pil_list = []
                        for b64 in frontier_imgs_b64:
                            try:
                                img_bytes = base64.b64decode(b64)
                                pil_list.append(_Image.open(_BytesIO(img_bytes)).convert('RGB'))
                            except Exception:
                                pil_list.append(_Image.new('RGB', (224, 224), color=(0, 0, 0)))
                        enc = store['png']['enc'] or {}
                        clip_model = str(enc.get('clip_model', 'ViT-B-32'))
                        pretrained = enc.get('open_clip_pretrained') or None
                        t_emb = _embed_images_with_clip(pil_list, model_name=clip_model, pretrained=pretrained, device=dev)
                        if t_emb is None:
                            logging.info("[VecRetrieve] image embedding failed")
                        else:
                            import numpy as _np
                            merged_candidates = []
                            # 日志：记录图片相似度检索结果
                            logging.info(f"[VecRetrieve][question-first] Image similarity retrieval for {len(frontier_imgs_b64)} frontier(s):")
                            for i in range(t_emb.shape[0]):
                                v = t_emb[i]
                                sims = corpus @ v.astype(_np.float32)
                                top_idx = _np.argsort(-sims)[: min(2000, max(50, top_k * 50))]
                                logging.info(f"  Frontier #{i} top image similarities:")
                                for rank, idx in enumerate(top_idx[:10], 1):  # 只显示前10个
                                    sim_score = float(sims[int(idx)])
                                    meta = meta_list[int(idx)]
                                    qid = meta.get('question_id')
                                    filename = meta.get('src_rel_path')
                                    logging.info(f"    #{rank}: clip_sim={sim_score:.4f} | qid={qid} | filename={filename}")
                                per_scores = []
                                for idx in top_idx:
                                    meta = meta_list[int(idx)]
                                    qid = meta.get('question_id')
                                    if exclude_question_id and qid == exclude_question_id:
                                        continue
                                    per_scores.append({
                                        'similarity': float(sims[int(idx)]),
                                        'question_id': qid,
                                        'step_key': meta.get('step_key'),
                                        'level': meta.get('level'),
                                        'filename_rel': meta.get('src_rel_path'),
                                        'source_frontier_index': i,
                                    })
                                seen = set()
                                kept = []
                                for cand in per_scores:
                                    key = (cand['question_id'], cand.get('step_key'))
                                    if key in seen:
                                        continue
                                    seen.add(key)
                                    kept.append(cand)
                                    if len(kept) >= 5:
                                        break
                                merged_candidates.extend(kept)
                            if merged_candidates:
                                best_by_key = {}
                                for cand in merged_candidates:
                                    key = (cand['question_id'], cand.get('step_key'))
                                    prev = best_by_key.get(key)
                                    if prev is None or float(cand['similarity']) > float(prev['similarity']):
                                        best_by_key[key] = cand
                                merged_unique = list(best_by_key.values())

                                # question-first模式：直接按clip相似度排序，不用RRF
                                final_sorted = sorted(merged_unique, key=lambda x: x['similarity'], reverse=True)
                                try:
                                    log_n = min(max(1, top_k), len(final_sorted))
                                    if log_n > 0:
                                        logging.info(f"[VecRetrieve] question-first ranking by CLIP similarity (top {log_n}):")
                                    for rank, cand in enumerate(final_sorted[:log_n]):
                                        logging.info(
                                            f"  #{rank+1}: clip_sim={cand.get('similarity', 0.0):.4f} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')} | filename={cand.get('filename_rel') or 'N/A'}"
                                        )
                                except Exception:
                                    pass
                                final_selected = final_sorted[: max(1, top_k)]
                                exp = _load_experience(cfg)
                                texts = []
                                for j, cand in enumerate(final_selected):
                                    qid = cand.get('question_id')
                                    sk = cand.get('step_key')
                                    exp_text = None
                                    try:
                                        for ep_id, qdict in exp.items():
                                            if not isinstance(qdict, dict):
                                                continue
                                            qinfo = qdict.get(qid)
                                            if not isinstance(qinfo, dict):
                                                continue
                                            steps = qinfo.get('steps', {})
                                            if not isinstance(steps, dict):
                                                continue
                                            step_entry = steps.get(sk, {})
                                            if isinstance(step_entry, dict) and isinstance(step_entry.get('experience'), str):
                                                exp_text = step_entry['experience']
                                                break
                                        if isinstance(exp_text, str) and exp_text.strip():
                                            texts.append(f"Experience {j}: " + exp_text.strip())
                                    except Exception:
                                        continue
                                return "\n\n".join(texts) if texts else None
            except Exception as e:
                logging.warning(f"[VecRetrieve] question-first failed: {e}")

    if (strategy or 'sim') == 'sim':
        store = _load_vector_store(cfg)
        if not store or not frontier_imgs_b64:
            logging.info("[VecRetrieve] vector store not available or no frontiers")
        else:
            try:
                dev = _device_auto()
                from PIL import Image as _Image
                from io import BytesIO as _BytesIO
                pil_list = []
                for b64 in frontier_imgs_b64:
                    try:
                        img_bytes = base64.b64decode(b64)
                        pil_list.append(_Image.open(_BytesIO(img_bytes)).convert('RGB'))
                    except Exception:
                        pil_list.append(_Image.new('RGB', (224, 224), color=(0, 0, 0)))
                enc = store['png']['enc'] or {}
                clip_model = str(enc.get('clip_model', 'ViT-B-32'))
                pretrained = enc.get('open_clip_pretrained') or None
                t_emb = _embed_images_with_clip(pil_list, model_name=clip_model, pretrained=pretrained, device=dev)
                if t_emb is None:
                    logging.info("[VecRetrieve] image embedding failed; skip vecimg sim")
                else:
                    import numpy as _np
                    corpus = store['png']['emb']  # [N, D], 已归一化
                    merged_candidates = []
                    for i in range(t_emb.shape[0]):
                        v = t_emb[i]
                        sims = corpus @ v.astype(_np.float32)
                        # 取前若干高相似度的条目
                        top_idx = _np.argsort(-sims)[: min(2000, max(50, top_k * 50))]
                        per_scores = []
                        for idx in top_idx:
                            meta = store['png']['meta'][int(idx)]
                            qid = meta.get('question_id')
                            if exclude_question_id and qid == exclude_question_id:
                                continue
                            per_scores.append({
                                'similarity': float(sims[int(idx)]),
                                'question_id': qid,
                                'step_key': meta.get('step_key'),
                                'level': meta.get('level'),
                                'filename_rel': meta.get('src_rel_path'),
                                'source_frontier_index': i,
                            })
                        # 去重 top-5（按 (qid, step_key)）
                        seen = set()
                        kept = []
                        for cand in per_scores:
                            key = (cand['question_id'], cand['step_key'])
                            if key in seen:
                                continue
                            seen.add(key)
                            kept.append(cand)
                            if len(kept) >= 5:
                                break
                        merged_candidates.extend(kept)
                    if merged_candidates:
                        # 全局去重
                        best_by_key = {}
                        for cand in merged_candidates:
                            key = (cand['question_id'], cand['step_key'])
                            prev = best_by_key.get(key)
                            if prev is None or float(cand['similarity']) > float(prev['similarity']):
                                best_by_key[key] = cand
                        merged_unique = list(best_by_key.values())
                        # 文本相似（问题 → 训练集问题文本）
                        qid2question = _load_questions_en()
                        image_sorted = sorted(merged_unique, key=lambda x: x['similarity'], reverse=True)
                        text_scores = {}
                        if isinstance(current_question, str) and current_question.strip():
                            for cand in merged_unique:
                                qid = cand.get('question_id')
                                qtext = qid2question.get(qid, '')
                                text_scores[(cand.get('question_id'), cand.get('step_key'))] = _cosine_sim_tokens(current_question, qtext)
                        else:
                            for cand in merged_unique:
                                text_scores[(cand.get('question_id'), cand.get('step_key'))] = 0.0
                        text_sorted = sorted(merged_unique, key=lambda x: text_scores[(x.get('question_id'), x.get('step_key'))], reverse=True)
                        # RRF 融合
                        rank_img = { (c.get('question_id'), c.get('step_key')): ii+1 for ii, c in enumerate(image_sorted) }
                        rank_txt = { (c.get('question_id'), c.get('step_key')): ii+1 for ii, c in enumerate(text_sorted) }
                        for cand in merged_unique:
                            key = (cand.get('question_id'), cand.get('step_key'))
                            ri = rank_img.get(key, len(image_sorted) + 1)
                            rt = rank_txt.get(key, len(text_sorted) + 1)
                            cand['rrf_score'] = (1.0 / (rrf_k + ri)) + (1.0 / (rrf_k + rt))
                            cand['qsim'] = text_scores.get(key, 0.0)
                        final_sorted = sorted(merged_unique, key=lambda x: x.get('rrf_score', 0.0), reverse=True)
                        # 日志：展示 CLIP 相似度
                        try:
                            log_n = min(max(1, top_k), len(final_sorted))
                            if log_n > 0:
                                logging.info(f"[VecRetrieve] fused ranking by RRF (top {log_n}):")
                            for rank, cand in enumerate(final_sorted[:log_n]):
                                logging.info(
                                    f"  #{rank+1}: rrf={cand.get('rrf_score', 0.0):.6f} | clip_sim={cand.get('similarity', 0.0):.4f} | qsim={cand.get('qsim', 0.0):.4f} | src_idx={cand.get('source_frontier_index')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')}"
                                )
                        except Exception:
                            pass
                        # 选取 top_k 并回取经验
                        final_selected = final_sorted[: max(1, top_k)]
                        exp = _load_experience(cfg)
                        texts = []
                        for j, cand in enumerate(final_selected):
                            qid = cand.get('question_id')
                            sk = cand.get('step_key')
                            exp_text = None
                            try:
                                for ep_id, qdict in exp.items():
                                    if not isinstance(qdict, dict):
                                        continue
                                    qinfo = qdict.get(qid)
                                    if not isinstance(qinfo, dict):
                                        continue
                                    steps = qinfo.get('steps', {})
                                    if not isinstance(steps, dict):
                                        continue
                                    step_entry = steps.get(sk, {})
                                    if isinstance(step_entry, dict) and isinstance(step_entry.get('experience'), str):
                                        exp_text = step_entry['experience']
                                        break
                                if isinstance(exp_text, str) and exp_text.strip():
                                    texts.append(f"Experience {j}: " + exp_text.strip())
                            except Exception:
                                continue
                        return "\n\n".join(texts) if texts else None
            except Exception as e:
                logging.warning(f"[VecRetrieve] sim(vecimg) branch failed: {e}")

    # random：从向量库 PNG 通道随机抽取，映射到 (qid, step_key) 后回取 experience
    if (strategy or 'sim') == 'random':
        store = _load_vector_store(cfg)
        if not store:
            logging.info("[VecRetrieve][random] vector store not available")
        else:
            try:
                import random as _rnd
                meta = store['png']['meta'] or []
                # 随机打乱并去重 (qid, step_key)
                idxs = list(range(len(meta)))
                _rnd.shuffle(idxs)
                seen = set()
                selected = []
                for idx in idxs:
                    m = meta[idx]
                    qid = m.get('question_id')
                    sk = m.get('step_key')
                    if not qid or not sk:
                        continue
                    if exclude_question_id and qid == exclude_question_id:
                        continue
                    key = (qid, sk)
                    if key in seen:
                        continue
                    seen.add(key)
                    selected.append({'question_id': qid, 'step_key': sk, 'level': m.get('level')})
                    if len(selected) >= max(1, top_k):
                        break
                # 回取 experience
                exp = _load_experience(cfg)
                texts = []
                for j, cand in enumerate(selected):
                    qid = cand.get('question_id')
                    sk = cand.get('step_key')
                    exp_text = None
                    try:
                        for ep_id, qdict in exp.items():
                            if not isinstance(qdict, dict):
                                continue
                            qinfo = qdict.get(qid)
                            if not isinstance(qinfo, dict):
                                continue
                            steps = qinfo.get('steps', {})
                            if not isinstance(steps, dict):
                                continue
                            step_entry = steps.get(sk, {})
                            if isinstance(step_entry, dict) and isinstance(step_entry.get('experience'), str):
                                exp_text = step_entry['experience']
                                break
                        if isinstance(exp_text, str) and exp_text.strip():
                            texts.append(f"Experience {j}: " + exp_text.strip())
                    except Exception:
                        continue
                return "\n\n".join(texts) if texts else None
            except Exception as e:
                logging.warning(f"[VecRetrieve][random] failed: {e}")

    # 没有其他可用策略，返回 None
        return None

    # 预解码：索引 bits
    idx_bits = {}
    idx_meta = {}
    for rel_key, rec in index_map.items():
        if not isinstance(rec, dict):
            continue
        qid = rec.get('question_id')
        sk = rec.get('step_key')
        if not qid or not sk:
            continue
        if exclude_question_id and qid == exclude_question_id:
            continue
        bits_list = rec.get('bits')
        if not isinstance(bits_list, list):
            continue
        try:
            bits = np.asarray(bits_list, dtype=np.uint8).reshape(-1)
        except Exception:
            continue
        idx_bits[rel_key] = bits
        idx_meta[rel_key] = {
            'episode_id': None,            # 不依赖 episode 过滤，统一从 experience 里回取
            'question_id': qid,
            'step_key': sk,
            'level': rec.get('level'),
            'filename_rel': os.path.join('frontier', rec.get('filename', '')),
        }

    # 预解码：目标 bits
    target_bits_list = []
    for b64 in frontier_imgs_b64:
        bits = _pil_bits_from_b64(b64)
        if bits is not None:
            target_bits_list.append(bits)
    if not target_bits_list:
        return None

    # per-frontier：取去重后的 top-5 (qid, step_key)
    merged_candidates = []
    for i, tbits in enumerate(target_bits_list):
        # 对全索引计算最大相似度（与所有目标 bits 中的最大）
        per_scores = []
        for rel_key, cbits in idx_bits.items():
            try:
                sim = float(_similarity(cbits, tbits))
            except Exception:
                continue
            meta = idx_meta[rel_key]
            per_scores.append({
                'similarity': sim,
                **meta,
                'source_frontier_index': i,
            })
        if not per_scores:
            continue
        per_scores.sort(key=lambda x: x['similarity'], reverse=True)
        # 去重 top-5
        seen = set()
        kept = []
        for cand in per_scores:
            key = (cand['question_id'], cand['step_key'])
            if key in seen:
                continue
            seen.add(key)
            kept.append(cand)
            if len(kept) >= 5:
                break
        merged_candidates.extend(kept)

    if not merged_candidates:
        return None

    # 全局去重（按 (qid, step) 保留相似度最高的图像分数）
    best_by_key = {}
    for cand in merged_candidates:
        key = (cand['question_id'], cand['step_key'])
        prev = best_by_key.get(key)
        if prev is None or float(cand['similarity']) > float(prev['similarity']):
            best_by_key[key] = cand
    merged_unique = list(best_by_key.values())
    # 基于英文问题文本计算文本相似度
    qid2question = _load_questions_en()
    image_sorted = sorted(merged_unique, key=lambda x: x['similarity'], reverse=True)
    text_scores = {}
    if isinstance(current_question, str) and current_question.strip():
        for cand in merged_unique:
            qid = cand.get('question_id')
            qtext = qid2question.get(qid, '')
            text_scores[(cand.get('question_id'), cand.get('step_key'))] = _cosine_sim_tokens(current_question, qtext)
    else:
        for cand in merged_unique:
            text_scores[(cand.get('question_id'), cand.get('step_key'))] = 0.0

    text_sorted = sorted(merged_unique, key=lambda x: text_scores[(x.get('question_id'), x.get('step_key'))], reverse=True)
    # RRF 前的统计日志
    try:
        qtext_coverage = sum(1 for cand in merged_unique if qid2question.get(cand.get('question_id')))
        qsim_values = [text_scores[(c.get('question_id'), c.get('step_key'))] for c in merged_unique]
        nonzero_qsim = [v for v in qsim_values if v > 0]
        nz_count = len(nonzero_qsim)
        nz_mean = (sum(nonzero_qsim) / nz_count) if nz_count > 0 else 0.0
        nz_max = max(nonzero_qsim) if nz_count > 0 else 0.0
        logging.info(
            f"[SimpleRecall][RRF] candidates={len(merged_unique)} | qtext_coverage={qtext_coverage} | nonzero_qsim={nz_count} | qsim_mean={nz_mean:.4f} | qsim_max={nz_max:.4f} | rrf_k={rrf_k}"
        )
        # 各自通道的前 top_k 摘要
        log_n = min(max(1, top_k), len(image_sorted))
        if log_n > 0:
            logging.info(f"[SimpleRecall][RRF] image-only ranking (top {log_n}):")
            for rank, cand in enumerate(image_sorted[:log_n]):
                logging.info(
                    f"  #{rank+1}: img_sim={cand.get('similarity', 0.0):.4f} | qid={cand.get('question_id')} | step={cand.get('step_key')}"
                )
            logging.info(f"[SimpleRecall][RRF] text-only ranking (top {log_n}):")
            for rank, cand in enumerate(text_sorted[:log_n]):
                key = (cand.get('question_id'), cand.get('step_key'))
                logging.info(
                    f"  #{rank+1}: qsim={text_scores.get(key, 0.0):.4f} | qid={cand.get('question_id')} | step={cand.get('step_key')}"
                )
    except Exception:
        pass
    # 计算 RRF 分数：score = 1/(k + rank_img) + 1/(k + rank_txt)
    rank_img = { (c.get('question_id'), c.get('step_key')): i+1 for i, c in enumerate(image_sorted) }
    rank_txt = { (c.get('question_id'), c.get('step_key')): i+1 for i, c in enumerate(text_sorted) }
    for cand in merged_unique:
        key = (cand.get('question_id'), cand.get('step_key'))
        ri = rank_img.get(key, len(image_sorted) + 1)
        rt = rank_txt.get(key, len(text_sorted) + 1)
        cand['rrf_score'] = (1.0 / (rrf_k + ri)) + (1.0 / (rrf_k + rt))
        cand['qsim'] = text_scores.get(key, 0.0)
    # 打印融合后排行（仅 top_k）
    final_sorted = sorted(merged_unique, key=lambda x: x.get('rrf_score', 0.0), reverse=True)
    try:
        logging.info(f"[SimpleRecall] final candidate pool size={len(merged_unique)}, strategy={strategy}, request_top_k={top_k}")
        log_n = min(max(1, top_k), len(final_sorted))
        if log_n > 0:
            logging.info(f"[SimpleRecall] fused ranking by RRF (top {log_n}):")
        for rank, cand in enumerate(final_sorted[:log_n]):
            logging.info(
                f"  #{rank+1}: rrf={cand.get('rrf_score', 0.0):.6f} | img_sim={cand.get('similarity', 0.0):.4f} | qsim={cand.get('qsim', 0.0):.4f} | src_idx={cand.get('source_frontier_index')} | qid={cand.get('question_id')} | step={cand.get('step_key')} | lvl={cand.get('level')}"
            )
    except Exception:
        pass

    if (strategy or 'sim') == 'random':
        k = min(max(1, top_k), len(merged_unique))
        final_selected = random.sample(merged_unique, k) if k > 0 else []
    else:
        final_selected = final_sorted[: max(1, top_k)]

    # 回取 experience 文本
    exp = _load_experience(cfg)
    texts = []
    for j, cand in enumerate(final_selected):
        qid = cand.get('question_id')
        sk = cand.get('step_key')
        exp_text = None
        # 不知道 episode_id 时，穷举查找一次（字典层级通常不大）
        try:
            for ep_id, qdict in exp.items():
                if not isinstance(qdict, dict):
                    continue
                qinfo = qdict.get(qid)
                if not isinstance(qinfo, dict):
                    continue
                steps = qinfo.get('steps', {})
                if not isinstance(steps, dict):
                    continue
                step_entry = steps.get(sk, {})
                if isinstance(step_entry, dict) and isinstance(step_entry.get('experience'), str):
                    exp_text = step_entry['experience']
                    break
            if isinstance(exp_text, str) and exp_text.strip():
                texts.append(f"Experience {j}: " + exp_text.strip())
        except Exception:
            continue
        try:
            logging.info(
                f"[SimpleRecall] selected #{j+1}: rrf={cand.get('rrf_score', 0.0):.6f} | img_sim={cand.get('similarity', 0.0):.4f} | qsim={cand.get('qsim', 0.0):.4f} | src_idx={cand.get('source_frontier_index')} | qid={qid} | step={sk} | lvl={cand.get('level')}"
            )
        except Exception:
            continue

    return "\n\n".join(texts) if texts else None


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
    episodic_con=None,  # Episodic context text (this episode: recent steps/path & seen/unseen summary)
    frontier_type: str = "BVF",  # "BVF" for broad-view (layer0) or "CVF" for closer-view (layer1)
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
    # System role & definitions (based on user's template)
    # =========================
    label_word = "BVF" if str(frontier_type).upper() == "BVF" else "CVF"
    sys_prompt = (
        "You are an embodied agent for exploration in an indoor environment to answer a question. "
        "At each step of exploration, you will be given frontier snapshots of your surrounding environment; your task is to pick EXACTLY ONE frontier to move to for further exploration or solving the question.\n\n"
        "FRONTIERs are candidate entry points toward yet-unseen or information-rich regions—typical visual patterns include doorways/thresholds, corridors/intersections, "
        "stairs, corners/turns, or vantage points that likely open new coverage.\n\n"
        "You will be given 2 types of frontiers: Broad-View Frontier (BVF) segments your 360° surrounding environment so that you can have an overview. "
        "You SHALL pick EXACTLY ONE BVF to look closer. With the selected BVF, you DO NOT move; you further break down that direction into Closer-View Frontiers (CVF), which give narrowed perspectives. "
        "You SHALL pick EXACTLY ONE CVF to move to in the next step.\n\n"
        "You will also be given the following information as contexts:\n"
        "EGOCENTRIC VIEW (if shown): The agent’s immediate forward-looking camera view; use it as local context only.\n"
        "EPISODIC CONTEXT (if present): A factual textual summary of the previous steps within THIS episode (visited path, observations, likely-unseen areas). "
        "Use this to avoid redundancy and prefer novel, informative directions. It is evidence, not a command.\n"
        "EXPERIENCE REPLAY (if present): A textual experience of frontier selection to solve a similar question in a similar environment—how the decision was made, which frontier was chosen, what actions followed, the outcome/reward, a brief critique, and an abstraction to reflect on.\n\n"
        "RULES:\n"
        "- You will only be given either BVFs or CVFs at a time (BVF for looking closer; CVF for moving next).\n"
        "- Your reasoning must be concrete and visual. Name specific objects, layouts, textures, lighting, text-bearing surfaces/symbols, and any cues directly relevant to the question.\n"
        "- You must select one of the provided candidates; do NOT output that none is suitable.\n"
        f"- Output the rationale first and the answer last. On the final line, print ONLY '{label_word} i' (the chosen index).\n"
    )

    content = []

    # =========================
    # Frontier candidates
    # =========================
    content.append((f"You are given the following frontiers ({frontier_type} only at this step):",))
    if len(frontier_imgs) == 0:
        content.append(("No frontier is available.",))
    else:
        for i in range(len(frontier_imgs)):
            content.append((f"{label_word} {i}", frontier_imgs[i]))

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
            "Experience replay — knowledge from OTHER episodes in similar scenes. It may include 'Critique:' (what happened) and 'Abstraction:' (a simple rule). "
            "Use the Abstraction as a transferable hint for this scene. Extract only transferable visual patterns/strategies and prefer the current visible evidence when conflicts arise:\n"
            + context.strip(),
        ))

    # =========================
    # Question
    # =========================
    q_text = f"Now you need to answer the question: {question}"
    if image_goal is not None:
        content.append((q_text, image_goal))
    else:
        content.append((q_text,))

    # =========================
    # Minimal reasoning scaffold consistent with user's instruction
    # =========================
    guidance = (
        "IMPORTANT: You MUST reason step by step using ONLY the provided frontiers (Step 1, Step 2, ...), and do NOT skip steps. "
        "Use the contexts (EPISODIC/EXPERIENCE) if present to avoid redundancy and transfer useful cues. "
        f"Output the rationale first and the answer last. On the final line, print ONLY '{label_word} i'."
    )
    content.append((guidance,))

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
    # 支持 'frontier i'、'bvf i'、'cvf i' 三种格式（取最后一个命中）
    matches = list(re.finditer(r'(?:frontier|bvf|cvf)\s*(\d+)', output, re.IGNORECASE))
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
            full_response = call_openai_api(
                sys_prompt,
                content,
                seed=(int(getattr(cfg, "chat_seed")) if hasattr(cfg, "chat_seed") and getattr(cfg, "chat_seed") is not None else None)
            )
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
        # 使用本文件中的最简实现（不依赖 context_generator）
        step["replay_layer0_aggregated_context"] = simple_recall_and_aggregate(
            frontier_imgs_b64=frontier_imgs_0,
            cfg=cfg,
            exclude_question_id=step.get("question_id"),
            top_k=_replay_top,
            strategy=(
                "random" if getattr(cfg, "replay_mode", "sim") == "random" else (
                    "question-first" if getattr(cfg, "replay_mode", "sim") == "question-first" else "sim"
                )
            ),
            current_question=question,
        )
    else:
        step["replay_layer0_aggregated_context"] = None
        logging.info("[ReplayCtx] replay_top=0; skip layer0 recall and env context injection.")


    episodic_con = None
    if not os.path.exists(chosen_frontier_path):
        os.makedirs(chosen_frontier_path, exist_ok=True)

    # 允许通过 cfg.use_episodic_context 显式关闭 episodic context
    if bool(getattr(cfg, "use_episodic_context", True)):
        png_files = [f for f in os.listdir(chosen_frontier_path) if f.endswith('.png')]
        if len(png_files) > 0:
            sys_prompt, content = frontier_context(chosen_frontier_path)
            episodic_con = call_openai_api(
                sys_prompt,
                content,
                seed=(int(getattr(cfg, "chat_seed")) if hasattr(cfg, "chat_seed") and getattr(cfg, "chat_seed") is not None else None)
            )
            logging.info(f"Froncon label: {episodic_con}")
    else:
        logging.info("[EpisodicCtx] Disabled by cfg.use_episodic_context=False")



    # 当 _replay_top=0 时，不注入任何 experience（env recall）
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
        episodic_con=(episodic_con if bool(getattr(cfg, "use_episodic_context", True)) else None),
        frontier_type="BVF",
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
        full_response = call_openai_api(
            sys_prompt,
            content,
            seed=(int(getattr(cfg, "chat_seed")) if hasattr(cfg, "chat_seed") and getattr(cfg, "chat_seed") is not None else None)
        )
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
        idx_random = random.randrange(0, max(1, len(frontier_imgs_0)))
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
            # 根据选择的大簇子集做同样的最简实现
            subgroup_b64 = [frontier_imgs_1[i] for i in layer1_indices]
            layer1_context_text = simple_recall_and_aggregate(
                frontier_imgs_b64=subgroup_b64,
                cfg=cfg,
                exclude_question_id=step.get("question_id"),
                top_k=_replay_top,
                strategy=(
                    "random" if getattr(cfg, "replay_mode", "sim") == "random" else (
                        "question-first" if getattr(cfg, "replay_mode", "sim") == "question-first" else "sim"
                    )
                ),
                current_question=question,
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
            episodic_con=(episodic_con if bool(getattr(cfg, "use_episodic_context", True)) else None),
            frontier_type="CVF",
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
            full_response = call_openai_api(
                sys_prompt,
                content,
                seed=(int(getattr(cfg, "chat_seed")) if hasattr(cfg, "chat_seed") and getattr(cfg, "chat_seed") is not None else None)
            )
            
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
            idx_random = random.randrange(0, max(1, len(frontier_imgs_subgroup)))
            # 映射回全局 layer1 索引
            final_layer1_idx = layer1_indices[idx_random]
            global_frontier_idx = len(step["frontier_imgs_0"]) + final_layer1_idx
            response = f'frontier {global_frontier_idx}'
            reason = f"Randomly selected index {global_frontier_idx} due to parsing failure."
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

