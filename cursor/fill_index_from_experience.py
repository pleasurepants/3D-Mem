import os
import json
import argparse
from typing import Dict, Set, Tuple, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image
import numpy as np


def ahash_from_path(img_path: str, hash_size: int = 8) -> Optional[np.ndarray]:
    try:
        with Image.open(img_path) as im:
            im = im.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
        return (arr > arr.mean()).astype(np.uint8).reshape(-1)
    except Exception:
        return None


def bits_to_list(bits: np.ndarray) -> list:
    return [int(x) for x in list(bits.reshape(-1))]


def load_experience_steps(exp_path: str) -> Dict[str, Set[str]]:
    with open(exp_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    exp_steps = defaultdict(set)
    if isinstance(data, dict):
        for ep, qdict in data.items():
            if not isinstance(qdict, dict):
                continue
            for qid, qinfo in qdict.items():
                steps = (qinfo or {}).get('steps', {})
                if isinstance(steps, dict):
                    for sk in steps.keys():
                        exp_steps[qid].add(sk)
    return exp_steps


def load_index(index_path: str) -> Dict[str, dict]:
    if not os.path.exists(index_path):
        return {}
    with open(index_path, 'r', encoding='utf-8') as f:
        try:
            idx = json.load(f)
        except Exception:
            idx = {}
    return idx if isinstance(idx, dict) else {}


def collect_missing_targets(root: str, exp_steps: Dict[str, Set[str]], index_map: Dict[str, dict]) -> Dict[str, Set[str]]:
    """
    返回需要补齐的 {qid: {step_key}}：出现在 experience 中但索引里没有任何该 step 的 png 条目。
    判定：索引中存在 rel_key 形如 qid/frontier/*.png 且 rec.step_key == step_key 即视为存在。
    """
    have = defaultdict(set)
    for rel_key, rec in index_map.items():
        if not isinstance(rec, dict):
            continue
        qid = rec.get('question_id')
        sk = rec.get('step_key')
        if qid and sk:
            have[qid].add(sk)

    missing = defaultdict(set)
    for qid, sset in exp_steps.items():
        miss = set(sset) - have.get(qid, set())
        if miss:
            missing[qid].update(sorted(miss))
    return missing


def list_step_pngs(root: str, qid: str, step_key: str) -> Tuple[list, list]:
    """
    返回 (layer0_pngs, layer1_pngs) 的绝对路径列表。
    step_key: 'step_9' -> 匹配 '9-layer0-*.png' 和 '9-layer1-*.png'
    """
    try:
        step_idx = int(step_key.split('_')[1])
    except Exception:
        return [], []

    fr_dir = os.path.join(root, qid, 'frontier')
    if not os.path.isdir(fr_dir):
        return [], []

    l0, l1 = [], []
    prefix0 = f"{step_idx}-layer0-"
    prefix1 = f"{step_idx}-layer1-"
    for fn in os.listdir(fr_dir):
        if not fn.endswith('.png'):
            continue
        if fn.startswith(prefix0):
            l0.append(os.path.join(fr_dir, fn))
        elif fn.startswith(prefix1):
            l1.append(os.path.join(fr_dir, fn))
    return l0, l1


def build_records(root: str, qid: str, step_key: str, paths: list, level: str, hash_size: int) -> Dict[str, dict]:
    records = {}
    def _one(p: str):
        bits = ahash_from_path(p, hash_size)
        if bits is None:
            return None
        fn = os.path.basename(p)
        rel_key = os.path.join(qid, 'frontier', fn)
        rec = {
            'bits': bits_to_list(bits),
            'question_id': qid,
            'filename': fn,
            'step_key': step_key,
            'level': level,
        }
        return rel_key, rec

    with ThreadPoolExecutor(max_workers=16) as ex:
        futs = [ex.submit(_one, p) for p in paths]
        for fut in as_completed(futs):
            item = fut.result()
            if item:
                k, v = item
                records[k] = v
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='retrieve root, e.g., /.../qwen-exp-168')
    parser.add_argument('--exp', required=True, help='path to experience_output.json')
    parser.add_argument('--index', required=True, help='path to .frontier_ahash_index.json (will be updated unless --out set)')
    parser.add_argument('--out', default='', help='optional output path; if empty, overwrite index in-place')
    parser.add_argument('--hash_size', type=int, default=8)
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    exp_steps = load_experience_steps(args.exp)
    index_map = load_index(args.index)

    missing = collect_missing_targets(root, exp_steps, index_map)
    total_missing_pairs = sum(len(v) for v in missing.values())
    print(f"[Fill] qids_with_missing={len(missing)} missing_pairs(qid,step)={total_missing_pairs}")

    added = 0
    for qid, steps in sorted(missing.items()):
        for sk in sorted(steps):
            l0, l1 = list_step_pngs(root, qid, sk)
            if not l0 and not l1:
                print(f"[Fill] no frontier pngs for {qid} {sk}")
                continue
            recs0 = build_records(root, qid, sk, l0, 'layer0', args.hash_size)
            recs1 = build_records(root, qid, sk, l1, 'layer1', args.hash_size)
            for k, v in {**recs0, **recs1}.items():
                if k not in index_map:
                    index_map[k] = v
                    added += 1

    out_path = args.out if args.out else args.index
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(index_map, f, ensure_ascii=False, separators=(",", ":"))
    print(f"[Fill] added_entries={added} -> saved to {out_path}")


if __name__ == '__main__':
    main()


