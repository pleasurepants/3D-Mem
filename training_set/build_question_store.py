import os
import sys
import json
import argparse
import logging
from typing import List, Dict

import numpy as np


def load_questions_map(qpath: str) -> Dict[str, str]:
    try:
        with open(qpath, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        qmap: Dict[str, str] = {}
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    qid = it.get('question_id')
                    qtext = it.get('question')
                    if isinstance(qid, str) and isinstance(qtext, str) and qid and qtext:
                        qmap[qid] = qtext
        return qmap
    except Exception as e:
        logging.error(f"[QStore] load questions failed: {e}")
        return {}


def get_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def load_sbert(sbert_model: str, device: str):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(sbert_model, device=device)

        def _encode(texts: List[str]) -> np.ndarray:
            emb = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(emb, dtype=np.float32)

        dim = int(model.get_sentence_embedding_dimension())
        return _encode, dim
    except Exception as e:
        raise RuntimeError(f"Failed to load Sentence-BERT '{sbert_model}': {e}")


def build_faiss(vecs: np.ndarray, factory: str):
    try:
        import faiss
    except Exception as e:
        logging.warning(f"[QStore] faiss not available: {e}")
        return None
    d = vecs.shape[1]
    if factory.lower() == 'flat':
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.index_factory(d, factory)
    faiss.normalize_L2(vecs)
    index.add(vecs)
    return index


def save_faiss(index, path: str):
    if index is None:
        return
    import faiss
    faiss.write_index(index, path)


def main():
    parser = argparse.ArgumentParser(description='Build question-only vector store (SBERT) for retrieval.')
    parser.add_argument('--questions_path', type=str, required=True, help='绝对路径：aeqa_questions-168.json')
    parser.add_argument('--dst_root', type=str, required=True, help='输出根目录，如 .../qwen-exp-168/retrieve')
    parser.add_argument('--sbert_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--faiss_factory_txt', type=str, default='Flat')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    dst_root = os.path.abspath(args.dst_root)
    dst_question_root = os.path.join(dst_root, 'question')
    os.makedirs(dst_question_root, exist_ok=True)

    logging.info(f"[QStore] dst_root = {dst_root}")
    logging.info(f"[QStore] questions = {args.questions_path}")

    qmap = load_questions_map(args.questions_path)
    if not qmap:
        logging.error('[QStore] empty question map')
        sys.exit(2)

    # 排序保证稳定
    qids = sorted(qmap.keys())
    texts = [qmap[qid] for qid in qids]

    device = get_device()
    logging.info(f"[QStore] device = {device}")
    encode_text, dim_txt = load_sbert(args.sbert_model, device)
    logging.info(f"[QStore] start text embeddings for {len(texts)} questions (SBERT)")
    txt_emb = encode_text(texts)
    logging.info(f"[QStore] embeddings shape: {txt_emb.shape}")

    # 保存
    np.save(os.path.join(dst_question_root, 'embeddings.npy'), txt_emb)
    with open(os.path.join(dst_question_root, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump([{ 'question_id': qid, 'question': qmap[qid] } for qid in qids], f, ensure_ascii=False)
    with open(os.path.join(dst_question_root, 'encoders.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'text_encoder': 'sbert',
            'sbert_model': args.sbert_model,
            'dim_txt': int(dim_txt),
            'faiss_factory': args.faiss_factory_txt,
        }, f, ensure_ascii=False)

    try:
        idx_txt = build_faiss(txt_emb.copy(), args.faiss_factory_txt)
        if idx_txt is not None:
            save_faiss(idx_txt, os.path.join(dst_question_root, 'index.faiss'))
            logging.info('[QStore] question/index.faiss saved.')
    except Exception as e:
        logging.warning(f"[QStore] build FAISS failed: {e}")

    logging.info('[QStore] Done.')


if __name__ == '__main__':
    main()


