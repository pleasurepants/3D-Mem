#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, Tuple, List


NUMERIC_PNG_RE = re.compile(r"^\d+\.png$", re.IGNORECASE)


def list_dirnames(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    try:
        return sorted(
            [
                name
                for name in os.listdir(path)
                if os.path.isdir(os.path.join(path, name))
            ]
        )
    except Exception:
        return []


def count_numeric_pngs(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    try:
        return sum(
            1
            for name in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, name))
            and NUMERIC_PNG_RE.match(name) is not None
        )
    except Exception:
        return 0


def compare_frontier_video_counts(
    base_a: str, base_b: str
) -> Dict[str, Dict[str, object]]:
    """
    对比两个根目录下相同 question_id 的 frontier_video 文件夹中的 \d+.png 数量。
    返回结构:
      {
        question_id: {
          "count_first": int,
          "count_second": int,
          "first_less_than_second": bool,
          "diff": int  # 第二个减第一个，若第一个更多则为负数
        },
        ...
      }
    只对两边都存在的 question_id 进行比较。
    """
    qa_dirs = set(list_dirnames(base_a))
    qb_dirs = set(list_dirnames(base_b))
    common_ids = sorted(qa_dirs & qb_dirs)

    result: Dict[str, Dict[str, object]] = {}
    for qid in common_ids:
        fa = os.path.join(base_a, qid, "frontier_video")
        fb = os.path.join(base_b, qid, "frontier_video")
        ca = count_numeric_pngs(fa)
        cb = count_numeric_pngs(fb)
        diff = cb - ca  # “少几个”按需求：第一个比第二个多时，此键为负数

        result[qid] = {
            "count_first": ca,
            "count_second": cb,
            "first_less_than_second": ca < cb,
            "diff": diff,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比两个路径下相同 question_id 的 frontier_video 内 \d+.png 数量，并输出 JSON 报告。"
    )
    parser.add_argument(
        "first_dir",
        type=str,
        default="/anvme/.../unformat-all-sim-t5-s568",
        help="第一个根目录（例如：/anvme/.../unformat-all-sim-t5-s568）",
    )
    parser.add_argument(
        "second_dir",
        type=str,
        default="/anvme/.../hiera-cot-seed568",
        help="第二个根目录（例如：/anvme/.../hiera-cot-seed568）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="输出 JSON 路径（可选）。默认输出到 /home/hpc/v100dd/v100dd12/code/3D-Mem/teaser/compare_frontier_video_{ts}.json",
    )
    args = parser.parse_args()

    base_a = os.path.abspath(args.first_dir)
    base_b = os.path.abspath(args.second_dir)

    report = compare_frontier_video_counts(base_a, base_b)

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = "/home/hpc/v100dd/v100dd12/code/3D-Mem/teaser"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"compare_frontier_video_{ts}.json")

    payload = {
        "first_dir": base_a,
        "second_dir": base_b,
        "summary": {
            "num_question_ids_compared": len(report),
            "num_first_less_than_second": sum(1 for v in report.values() if v["first_less_than_second"]),
        },
        "details": report,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(out_path)


if __name__ == "__main__":
    main()


