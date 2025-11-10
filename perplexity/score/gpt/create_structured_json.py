#!/usr/bin/env python3
"""
将GPT评分结果转换为结构化JSON，格式与traj_abs_format_score_structured.json一致。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from utils import DEFAULT_DATA_PATH, collect_records, dump_json, load_data

GROUP_NAMES = ("Low", "Medium", "High")


def ensure_minimum_samples(records: Sequence[Dict[str, float]]) -> None:
    if len(records) < 3:
        raise ValueError("生成结构化JSON至少需要3条有效评分记录。")


def sort_records(results: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    records = collect_records(results, extra_fields=("abstraction",))
    ensure_minimum_samples(records)
    records.sort(key=lambda item: (item["score"], item["id"]))
    return records


def create_groups(records: Sequence[Dict[str, float]], split1: int, split2: int) -> Dict[str, List[Dict[str, float]]]:
    n = len(records)
    if split1 <= 0 or split2 <= split1 or split2 >= n:
        raise ValueError("分割索引不合法。")

    low = list(records[:split1])
    mid = list(records[split1:split2])
    high = list(records[split2:])

    if not low or not mid or not high:
        raise ValueError("分组结果存在空组。")

    return {"Low": low, "Medium": mid, "High": high}


def evaluate_groups(groups: Dict[str, List[Dict[str, float]]]) -> Tuple[float, float, float]:
    sizes = np.asarray([len(groups[name]) for name in GROUP_NAMES], dtype=float)
    variance = float(np.var(sizes))

    means = []
    for name in GROUP_NAMES:
        scores = np.asarray([item["score"] for item in groups[name]], dtype=float)
        means.append(float(np.mean(scores)))

    separation = (means[1] - means[0]) + (means[2] - means[1])
    objective = 0.7 * variance - 0.3 * separation
    return variance, separation, objective


def find_balanced_groups(records: Sequence[Dict[str, float]]) -> Tuple[Dict[str, List[Dict[str, float]]], float, float]:
    n = len(records)
    best_groups: Dict[str, List[Dict[str, float]]] | None = None
    best_objective: float | None = None
    best_boundaries: Tuple[float, float] | None = None

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            try:
                groups = create_groups(records, i, j)
            except ValueError:
                continue

            variance, separation, objective = evaluate_groups(groups)

            if best_objective is None or objective < best_objective:
                best_groups = groups
                best_objective = objective
                low_boundary = groups["Low"][-1]["score"]
                mid_boundary = groups["Medium"][-1]["score"]
                best_boundaries = (low_boundary, mid_boundary)

    if best_groups is None or best_boundaries is None:
        raise RuntimeError("未能找到合适的平衡分组方案。")

    return best_groups, best_boundaries[0], best_boundaries[1]


def build_summary(
    total_samples: int,
    groups: Dict[str, List[Dict[str, float]]],
    low_boundary: float,
    high_boundary: float,
    source_file: Path,
) -> Dict[str, object]:
    distribution = {}
    for name in GROUP_NAMES:
        scores = np.asarray([item["score"] for item in groups[name]], dtype=float)
        distribution[name] = {
            "count": int(scores.size),
            "mean_score": float(np.mean(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
        }

    grouping_criteria = {
        "Low": f"< {low_boundary:.3f}",
        "Medium": f"{low_boundary:.3f} - {high_boundary:.3f}",
        "High": f">= {high_boundary:.3f}",
    }

    return {
        "total_samples": total_samples,
        "rank_distribution": distribution,
        "created_at": datetime.now().isoformat(),
        "source_file": str(source_file),
        "grouping_criteria": grouping_criteria,
    }


def convert_to_structured(
    groups: Dict[str, List[Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, object]]]:
    structured = {}
    for name in GROUP_NAMES:
        group_payload: Dict[str, Dict[str, object]] = {}
        for record in groups[name]:
            group_payload[record["id"]] = {
                "abstraction": record.get("abstraction", ""),
                "average_score": record["score"],
            }
        structured[name] = group_payload
    return structured


def create_structured_json(input_path: Path, output_path: Path) -> None:
    _, results = load_data(input_path)
    records = sort_records(results)
    groups, low_boundary, high_boundary = find_balanced_groups(records)

    summary = build_summary(len(records), groups, low_boundary, high_boundary, input_path)
    structured_groups = convert_to_structured(groups)

    payload: Dict[str, object] = {"summary": summary}
    payload.update(structured_groups)

    dump_json(payload, output_path)
    print(f"结构化JSON已生成: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成GPT评分的结构化JSON文件。")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="输入的GPT评分JSON路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "gpt_score_structured.json",
        help="输出结构化JSON路径。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_structured_json(args.input, args.output)


if __name__ == "__main__":
    main()

