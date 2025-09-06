import os
import json
import time
import argparse
from typing import Dict, Tuple, Optional

from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


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


def parse_meta_from_filename(fn: str) -> Tuple[Optional[str], Optional[str]]:
    # infer step_key and level from filename like "0-layer0-1.png" or "0-layer1-1_2.png"
    try:
        prefix = fn.split('-', 1)[0]
        step_key = f"step_{int(prefix)}"
    except Exception:
        step_key = None
    level = None
    if "-layer0-" in fn:
        level = "layer0"
    elif "-layer1-" in fn:
        level = "layer1"
    return step_key, level


def build_index(root: str, workers: int, hash_size: int, merge: bool, output: Optional[str]) -> str:
    root = os.path.abspath(root)
    if output is None or len(output.strip()) == 0:
        output = os.path.join(root, ".frontier_ahash_index.json")

    existing: Dict[str, Dict] = {}
    if merge and os.path.exists(output):
        try:
            with open(output, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}

    # collect all frontier pngs
    png_paths = []
    for qid in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        fr_dir = os.path.join(root, qid, "frontier")
        if not os.path.isdir(fr_dir):
            continue
        for fn in os.listdir(fr_dir):
            if not fn.endswith(".png"):
                continue
            png_paths.append((qid, os.path.join(fr_dir, fn), fn))

    start = time.time()
    updated = 0

    def _process(entry):
        qid, path, fn = entry
        # 兼容旧索引：若已有相同相对键且包含 bits，则跳过
        bits = ahash_from_path(path, hash_size)
        if bits is None:
            return None
        step_key, level = parse_meta_from_filename(fn)
        rel_key = os.path.join(qid, "frontier", fn)
        if rel_key in existing and isinstance(existing.get(rel_key, {}).get("bits"), list):
            return None
        return rel_key, {
            "bits": bits_to_list(bits),
            "question_id": qid,
            "filename": fn,
            "step_key": step_key,
            "level": level,
        }

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_process, e) for e in png_paths]
        for fut in as_completed(futures):
            item = fut.result()
            if not item:
                continue
            k, v = item
            existing[k] = v
            updated += 1

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        # 紧凑模式：无缩进、bits 列表单行
        json.dump(existing, f, ensure_ascii=False, separators=(",", ":"))

    elapsed = time.time() - start
    print(f"[Index] root={root}")
    print(f"[Index] images scanned={len(png_paths)}, updated={updated}, total={len(existing)}")
    print(f"[Index] output={output}")
    print(f"[Index] time={elapsed:.2f}s")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="retrieve root, e.g., /anvme/.../qwen-exp-168")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--hash_size", type=int, default=8)
    parser.add_argument("--merge", action="store_true", help="merge with existing index if present")
    parser.add_argument("--output", type=str, default="", help="output json path; default <root>/.frontier_ahash_index.json")
    args = parser.parse_args()

    out = build_index(
        root=args.root,
        workers=args.workers,
        hash_size=args.hash_size,
        merge=args.merge,
        output=(args.output if args.output else None),
    )
    print(out)


if __name__ == "__main__":
    main()


