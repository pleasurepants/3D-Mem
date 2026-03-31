import os
os.environ.setdefault("PILLOW_DISABLE_LERC", "1")
import sys
import json
import argparse
import logging
import shutil
from typing import List, Dict, Tuple

import numpy as np
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MAGNUM_LOG"] = "quiet"

def find_frontier_pngs(src_root: str) -> List[Dict]:
    items: List[Dict] = []
    for dirpath, dirnames, filenames in os.walk(src_root):
        base = os.path.basename(dirpath)
        if base.lower() != 'frontier':
            continue
        qid = os.path.basename(os.path.dirname(dirpath))
        for fn in filenames:
            if fn.lower().endswith('.png'):
                items.append({
                    'qid': qid,
                    'png_path': os.path.join(dirpath, fn)
                })
    return items


def load_frontier_index(src_root: str) -> Dict[Tuple[str, str], Dict]:
    idx_path = os.path.join(src_root, '.frontier_ahash_index.json')
    mapping = {}
    if not os.path.exists(idx_path):
        logging.warning(f"[BuildRetrieve] frontier index not found: {idx_path}")
        return mapping
    try:
        with open(idx_path, 'r', encoding='utf-8') as f:
            index_map = json.load(f)
        for rec in index_map.values():
            if not isinstance(rec, dict):
                continue
            qid = rec.get('question_id')
            fname = rec.get('filename')
            if not qid or not fname:
                continue
            mapping[(qid, os.path.basename(fname))] = rec
    except Exception as e:
        logging.warning(f"[BuildRetrieve] load frontier index failed: {e}")
    return mapping


def load_questions_map(qpath: str) -> Dict[str, str]:
    try:
        with open(qpath, 'r', encoding='utf-8') as f:
            arr = json.load(f)
        qmap = {}
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                qid = it.get('question_id')
                qtext = it.get('question')
                if isinstance(qid, str) and isinstance(qtext, str) and qid and qtext:
                    qmap[qid] = qtext
        return qmap
    except Exception as e:
        logging.warning(f"[BuildRetrieve] load questions failed: {e}")
        return {}


def get_device() -> str:
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def load_clip(clip_model: str, device: str, open_clip_pretrained: str = None, open_clip_tokenizer_model: str = None):
    try:
        from transformers import CLIPProcessor, CLIPModel
        import torch
        model = CLIPModel.from_pretrained(clip_model)
        processor = CLIPProcessor.from_pretrained(clip_model)
        model = model.to(device)
        model.eval()

        def _preprocess(pil_list):
            inputs = processor(images=pil_list, return_tensors='pt')
            return {k: v.to(device) for k, v in inputs.items()}

        @torch.no_grad()
        def _encode_img(inputs) -> np.ndarray:
            feats = model.get_image_features(**inputs)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feats.cpu().numpy().astype(np.float32)

        @torch.no_grad()
        def _encode_txt(texts: List[str]) -> np.ndarray:
            t_inputs = processor(text=texts, return_tensors='pt', padding=True, truncation=True)
            t_inputs = {k: v.to(device) for k, v in t_inputs.items()}
            feats = model.get_text_features(**t_inputs)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feats.cpu().numpy().astype(np.float32)

        dim = int(model.config.projection_dim)
        return _preprocess, _encode_img, _encode_txt, dim, dim
    except Exception as e:
        logging.info(f"[BuildRetrieve] transformers CLIP failed ({e}), try open_clip...")

    try:
        import open_clip
        import torch
        model_name = clip_model
        pretrained = open_clip_pretrained or 'openai'
        lower = clip_model.lower()
        if 'vit-h-14' in lower and (open_clip_pretrained is None):
            pretrained = 'laion2B-s32B-b79K'
            model_name = 'ViT-H-14'
        elif 'vit-b-32' in lower and (open_clip_pretrained is None):
            pretrained = 'openai'
            model_name = 'ViT-B-32'
        elif '/' in clip_model and 'vit-base-patch32' in lower:
            model_name = 'ViT-B-32'
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        model = model.to(device)
        model.eval()
        tokenizer_name = open_clip_tokenizer_model or model_name
        tokenizer = open_clip.get_tokenizer(tokenizer_name)

        def _preprocess(pil_list):
            import torch as _torch
            ims = [preprocess(img).unsqueeze(0) for img in pil_list]
            batch = _torch.cat(ims, dim=0).to(device)
            return batch

        @torch.no_grad()
        def _encode_img(batch) -> np.ndarray:
            feats = model.encode_image(batch)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feats.cpu().numpy().astype(np.float32)

        @torch.no_grad()
        def _encode_txt(texts: List[str]) -> np.ndarray:
            tokens = tokenizer(texts)
            tokens = tokens.to(device)
            feats = model.encode_text(tokens)
            feats = feats.float()
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            return feats.cpu().numpy().astype(np.float32)

        dim = int(model.visual.output_dim)
        return _preprocess, _encode_img, _encode_txt, dim, dim
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP: {e}")


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


def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def compute_clip_embeddings(paths: List[str], preprocess, encode) -> np.ndarray:
    from PIL import Image
    feats: List[np.ndarray] = []
    for pbatch in batched(paths, 64):
        pil_list = []
        for p in pbatch:
            try:
                pil_list.append(Image.open(p).convert('RGB'))
            except Exception:
                from PIL import Image as _Img
                pil_list.append(_Img.new('RGB', (224, 224), color=(0, 0, 0)))
        inputs = preprocess(pil_list)
        feats.append(encode(inputs))
    return np.concatenate(feats, axis=0).astype(np.float32)


def build_faiss(vecs: np.ndarray, factory: str):
    try:
        import faiss
    except Exception as e:
        logging.warning(f"[BuildRetrieve] faiss not available: {e}")
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
    parser = argparse.ArgumentParser(description='Traverse training set and build retrieve store (png + question).')
    parser.add_argument('--src_root', type=str, required=True, help='')
    parser.add_argument('--dst_root', type=str, required=True, help='')
    parser.add_argument('--clip_model', type=str, default='openai/clip-vit-base-patch32')
    parser.add_argument('--open_clip_pretrained', type=str, default=None, help='')
    parser.add_argument('--open_clip_tokenizer_model', type=str, default=None, help='')
    parser.add_argument('--sbert_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument('--text_encoder', type=str, default='clip', choices=['clip', 'sbert'], help='')
    parser.add_argument('--faiss_factory_img', type=str, default='Flat')
    parser.add_argument('--faiss_factory_txt', type=str, default='Flat')
    parser.add_argument('--copy_images', action='store_true', help='')
    parser.add_argument('--questions_path', type=str, default='data/aeqa_questions-168.json', help='')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    src_root = os.path.abspath(args.src_root)
    dst_root = os.path.abspath(args.dst_root)

    dst_png_root = os.path.join(dst_root, 'png')
    dst_question_root = os.path.join(dst_root, 'question')
    os.makedirs(dst_png_root, exist_ok=True)
    os.makedirs(dst_question_root, exist_ok=True)

    logging.info(f"[BuildRetrieve] src_root = {src_root}")
    logging.info(f"[BuildRetrieve] dst_root = {dst_root}")

    items = find_frontier_pngs(src_root)
    if not items:
        logging.error('no frontier PNGs found.')
        sys.exit(2)
    logging.info(f"[BuildRetrieve] frontier PNGs: {len(items)}")

    qid_fname_to_rec = load_frontier_index(src_root)

    img_paths: List[str] = [it['png_path'] for it in items]
    img_meta: List[Dict] = []
    for it in items:
        rel = os.path.relpath(it['png_path'], src_root)
        qid = it['qid']
        fname = os.path.basename(it['png_path'])
        rec = qid_fname_to_rec.get((qid, fname), {})
        img_meta.append({
            'question_id': qid,
            'src_rel_path': rel,
            'step_key': rec.get('step_key'),
            'level': rec.get('level'),
            'filename': rec.get('filename', fname),
        })

    if args.copy_images:
        copied_files: List[str] = []
        total = len(items)
        logging.info(f"[BuildRetrieve] copying PNGs -> {dst_png_root} (total={total})")
        for i, it in enumerate(items, 1):
            qid = it['qid']
            src = it['png_path']
            new_name = f"{qid}__{os.path.basename(src)}"
            dst = os.path.join(dst_png_root, new_name)
            try:
                shutil.copy2(src, dst)
                copied_files.append(os.path.relpath(dst, dst_root))
            except Exception as e:
                logging.warning(f"copy failed: {src} -> {dst} ({e})")
            if (i % 500 == 0) or (i == total):
                logging.info(f"[BuildRetrieve] copied {i}/{total}")
        for m, rel_dst in zip(img_meta, copied_files):
            m['dst_rel_path'] = rel_dst

    device = get_device()
    logging.info(f"[BuildRetrieve] device = {device}")
    preprocess, encode_images, encode_text_clip, dim_img, dim_txt_clip = load_clip(
        args.clip_model, device, args.open_clip_pretrained, args.open_clip_tokenizer_model
    )
    if args.text_encoder == 'clip':
        encode_text = encode_text_clip
        dim_txt = dim_txt_clip
        text_encoder_tag = 'clip'
    else:
        encode_text, dim_txt = load_sbert(args.sbert_model, device)
        text_encoder_tag = 'sbert'

    logging.info(f"[BuildRetrieve] start CLIP image embeddings for {len(img_paths)} images (batch=64)")
    img_emb = []
    total_imgs = len(img_paths)
    processed = 0
    for pbatch in batched(img_paths, 64):
        from PIL import Image
        pil_list = []
        for p in pbatch:
            try:
                pil_list.append(Image.open(p).convert('RGB'))
            except Exception:
                from PIL import Image as _Img
                pil_list.append(_Img.new('RGB', (224, 224), color=(0, 0, 0)))
        inputs = preprocess(pil_list)
        batch_feats = encode_images(inputs)
        img_emb.append(batch_feats)
        for p in pbatch:
            processed += 1
            logging.info(f"[BuildRetrieve] encoded {processed}/{total_imgs} {os.path.relpath(p, src_root)}")
    img_emb = np.concatenate(img_emb, axis=0).astype(np.float32)
    logging.info(f"[BuildRetrieve] image embeddings: {img_emb.shape}")

    qmap = load_questions_map(args.questions_path)
    uniq_qids = sorted(list({it['qid'] for it in items}))
    q_texts: List[str] = [qmap.get(qid, '') for qid in uniq_qids]
    logging.info(f"[BuildRetrieve] start text embeddings for {len(q_texts)} unique questions using {('CLIP' if text_encoder_tag=='clip' else 'SBERT')}")
    txt_emb = encode_text(q_texts)
    logging.info(f"[BuildRetrieve] question embeddings: {txt_emb.shape} (unique qids={len(uniq_qids)})")

    np.save(os.path.join(dst_png_root, 'embeddings.npy'), img_emb)
    with open(os.path.join(dst_png_root, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(img_meta, f, ensure_ascii=False)
    with open(os.path.join(dst_png_root, 'encoders.json'), 'w', encoding='utf-8') as f:
        json.dump({
            'clip_model': args.clip_model,
            'open_clip_pretrained': args.open_clip_pretrained,
            'open_clip_tokenizer_model': args.open_clip_tokenizer_model,
            'dim_img': int(dim_img),
            'faiss_factory': args.faiss_factory_img,
        }, f, ensure_ascii=False)
    try:
        idx_img = build_faiss(img_emb.copy(), args.faiss_factory_img)
        if idx_img is not None:
            save_faiss(idx_img, os.path.join(dst_png_root, 'index.faiss'))
            logging.info('[BuildRetrieve] png/index.faiss saved.')
    except Exception as e:
        logging.warning(f"build image FAISS failed: {e}")

    np.save(os.path.join(dst_question_root, 'embeddings.npy'), txt_emb)
    with open(os.path.join(dst_question_root, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump([{ 'question_id': qid, 'question': qmap.get(qid, '') } for qid in uniq_qids], f, ensure_ascii=False)
    with open(os.path.join(dst_question_root, 'encoders.json'), 'w', encoding='utf-8') as f:
        if text_encoder_tag == 'clip':
            json.dump({
                'text_encoder': 'clip',
                'clip_model': args.clip_model,
                'dim_txt': int(dim_txt),
                'faiss_factory': args.faiss_factory_txt,
            }, f, ensure_ascii=False)
        else:
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
            logging.info('[BuildRetrieve] question/index.faiss saved.')
    except Exception as e:
        logging.warning(f"build question FAISS failed: {e}")

    logging.info('[BuildRetrieve] Done.')


if __name__ == '__main__':
    main()


