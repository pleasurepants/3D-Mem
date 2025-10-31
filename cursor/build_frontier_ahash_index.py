import os
import json
import time
import argparse
from typing import Dict, Tuple, Optional

from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

import re

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


def parse_meta_from_filename(fn: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # infer step_key and level from filename like "0-layer0-1.png" or "0-layer1-1_2.png"
    # 'task-0_step-34-layer1-1_1.png', 'task-0_step-34-layer0-1.png'
    step_key = None
    task_id = None
    try:
        # prefix = fn.split('-', 1)[0]
        # step_key = f"step_{int(prefix)}"
        m = re.match(r"task-(\d+)_step-(\d+)-layer(\d+)-(\d+)(?:_(\d+))?\.png$", fn)
        if m:
            step_key = f"step_{int(m.group(2))}"
            task_id = int(m.group(1))
    except Exception:
        pass
    level = None
    if "-layer0-" in fn:
        level = "layer0"
    elif "-layer1-" in fn:
        level = "layer1"
    return step_key, level, task_id


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
    print(f"[Index] Scanning for frontier PNG files in {root}...")
    png_paths = []
    for qid in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        fr_dir = os.path.join(root, qid, "frontier")
        if not os.path.isdir(fr_dir):
            continue
        for fn in os.listdir(fr_dir):
            if not fn.endswith(".png"):
                continue
            png_paths.append((qid, os.path.join(fr_dir, fn), fn))
            # '00016-qk9eeNeR4vw_ep_0', 
            # '/nfs/data8/jingpei/eqa/3D-Mem/results/goatbench/train_hierarchical-cot/seed13/00016-qk9eeNeR4vw_ep_0/frontier', 
            # 'task-0_step-34-layer1-1_1.png'

    start = time.time()
    updated = 0
    skipped = 0
    failed = 0
    processed = 0

    def _process(entry):
        qid, path, fn = entry
        # 兼容旧索引：若已有相同相对键且包含 bits，则跳过
        bits = ahash_from_path(path, hash_size)
        if bits is None:
            return None, "failed"
        step_key, level, task_id = parse_meta_from_filename(fn)
        rel_key = os.path.join(qid, "frontier", fn)
        if rel_key in existing and isinstance(existing.get(rel_key, {}).get("bits"), list):
            return None, "skipped"
        
        # 尝试解析 question_id
        try:
            if "_ep_" in qid:
                parts = qid.split("_ep_")
                scene_id = parts[0]
                episode_id = f"ep_{parts[1]}"
            else:
                # fallback: 使用原始 qid
                scene_id = qid
                episode_id = ""
            
            if task_id is not None:
                question_id = f"{scene_id}_{episode_id}_{task_id}"
            else:
                question_id = qid
        except Exception:
            question_id = qid
        
        return (rel_key, {
            "bits": bits_to_list(bits),
            "question_id": question_id,
            "filename": fn,
            "step_key": step_key,
            "level": level,
        }), "success"

    print(f"[Index] Processing {len(png_paths)} images with {workers} workers...")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(_process, e) for e in png_paths]
        for i, fut in enumerate(as_completed(futures), 1):
            result, status = fut.result()
            if status == "success" and result:
                k, v = result
                existing[k] = v
                updated += 1
            elif status == "skipped":
                skipped += 1
            elif status == "failed":
                failed += 1
            
            processed += 1
            # 每处理 100 个文件或每 10% 显示一次进度
            if processed % max(1, len(png_paths) // 10) == 0 or processed % 100 == 0 or processed == len(png_paths):
                elapsed_now = time.time() - start
                rate = processed / elapsed_now if elapsed_now > 0 else 0
                eta = (len(png_paths) - processed) / rate if rate > 0 else 0
                print(f"[Index] Progress: {processed}/{len(png_paths)} ({100*processed/len(png_paths):.1f}%) | "
                      f"Updated: {updated}, Skipped: {skipped}, Failed: {failed} | "
                      f"Speed: {rate:.1f} files/s | ETA: {eta:.1f}s")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        # 紧凑模式：无缩进、bits 列表单行
        json.dump(existing, f, ensure_ascii=False, separators=(",", ":"))

    elapsed = time.time() - start
    print(f"\n[Index] ========== Summary ==========")
    print(f"[Index] root={root}")
    print(f"[Index] images scanned={len(png_paths)}")
    print(f"[Index] updated={updated}, skipped={skipped}, failed={failed}")
    print(f"[Index] total entries in index={len(existing)}")
    print(f"[Index] output={output}")
    print(f"[Index] total time={elapsed:.2f}s ({elapsed/60:.2f} minutes)")
    if len(png_paths) > 0:
        print(f"[Index] average speed={len(png_paths)/elapsed:.2f} files/s")
    print(f"[Index] ==============================\n")
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


