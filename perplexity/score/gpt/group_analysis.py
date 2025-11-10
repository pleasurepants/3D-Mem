#!/usr/bin/env python3
"""
针对GPT评分结果的分组分析与可视化。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import DEFAULT_DATA_PATH, collect_records, dump_json, load_data

OUTPUT_DIR = Path(__file__).resolve().parent
GROUP_KEYS = ("low", "mid", "high")
GROUP_COLORS = {
    "low": "#e06666",
    "mid": "#ffd966",
    "high": "#93c47d",
}


@dataclass
class GroupingResult:
    name: str
    split_indices: Tuple[int, int]
    boundary_scores: Tuple[float, float]
    groups: Dict[str, List[Dict[str, float]]]
    balance_variance: float
    separation: float
    objective: float


def ensure_minimum_size(records: Sequence[Dict[str, float]]) -> None:
    if len(records) < 3:
        raise ValueError("进行三组划分至少需要3条有效评分记录。")


def sorted_records(results: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    records = collect_records(results, extra_fields=("question", "answer", "category"))
    ensure_minimum_size(records)
    records.sort(key=lambda r: (r["score"], r["id"]))
    return records


def create_groups_by_indices(
    records: Sequence[Dict[str, float]], split1: int, split2: int
) -> Dict[str, List[Dict[str, float]]]:
    n = len(records)
    split1 = max(1, min(split1, n - 2))
    split2 = max(split1 + 1, min(split2, n - 1))

    low_group = list(records[:split1])
    mid_group = list(records[split1:split2])
    high_group = list(records[split2:])

    if not low_group or not mid_group or not high_group:
        raise ValueError("划分结果存在空组，请检查分割索引。")

    return {"low": low_group, "mid": mid_group, "high": high_group}


def group_stats(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    scores = np.asarray([item["score"] for item in records], dtype=float)
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
    }


def evaluate_groups(groups: Dict[str, List[Dict[str, float]]]) -> Tuple[float, float, float]:
    sizes = np.asarray([len(groups[key]) for key in GROUP_KEYS], dtype=float)
    variance = float(np.var(sizes))

    means = []
    for key in GROUP_KEYS:
        scores = np.asarray([item["score"] for item in groups[key]], dtype=float)
        means.append(float(np.mean(scores)))

    separation = (means[1] - means[0]) + (means[2] - means[1])
    objective = 0.7 * variance - 0.3 * separation

    return variance, separation, objective


def build_strict_grouping(records: Sequence[Dict[str, float]]) -> GroupingResult:
    indices = np.array_split(np.arange(len(records)), 3)
    split1 = int(indices[0][-1]) + 1
    split2 = split1 + len(indices[1])
    groups = {
        "low": [records[i] for i in indices[0].tolist()],
        "mid": [records[i] for i in indices[1].tolist()],
        "high": [records[i] for i in indices[2].tolist()],
    }
    variance, separation, objective = evaluate_groups(groups)
    boundary_scores = (groups["low"][-1]["score"], groups["mid"][-1]["score"])
    return GroupingResult("strict", (split1, split2), boundary_scores, groups, variance, separation, objective)


def build_quantile_grouping(records: Sequence[Dict[str, float]]) -> GroupingResult:
    scores = np.asarray([item["score"] for item in records], dtype=float)
    q33, q67 = np.percentile(scores, [33.33, 66.67])

    low: List[Dict[str, float]] = []
    mid: List[Dict[str, float]] = []
    high: List[Dict[str, float]] = []

    for record in records:
        score = record["score"]
        if score < q33:
            low.append(record)
        elif score < q67:
            mid.append(record)
        else:
            high.append(record)

    if not low or not mid or not high:
        split1 = max(1, len(records) // 3)
        split2 = max(split1 + 1, 2 * len(records) // 3)
        groups = create_groups_by_indices(records, split1, split2)
        boundary_scores = (groups["low"][-1]["score"], groups["mid"][-1]["score"])
    else:
        groups = {"low": low, "mid": mid, "high": high}
        boundary_scores = (low[-1]["score"], mid[-1]["score"])

    split1 = len(groups["low"])
    split2 = split1 + len(groups["mid"])
    variance, separation, objective = evaluate_groups(groups)
    return GroupingResult("quantile", (split1, split2), boundary_scores, groups, variance, separation, objective)


def build_balanced_grouping(records: Sequence[Dict[str, float]]) -> GroupingResult:
    n = len(records)
    best_result: GroupingResult | None = None

    for i in range(1, n - 1):
        for j in range(i + 1, n):
            try:
                groups = create_groups_by_indices(records, i, j)
            except ValueError:
                continue
            variance, separation, objective = evaluate_groups(groups)
            boundary_scores = (groups["low"][-1]["score"], groups["mid"][-1]["score"])

            candidate = GroupingResult(
                "balanced",
                (i, j),
                boundary_scores,
                groups,
                variance,
                separation,
                objective,
            )

            if best_result is None or candidate.objective < best_result.objective:
                best_result = candidate

    if best_result is None:
        raise RuntimeError("未能找到满足条件的平衡分组方案。")

    return best_result


def summarize_method(result: GroupingResult) -> Dict[str, Dict[str, float]]:
    return {
        "split_indices": {"low_mid": result.split_indices[0], "mid_high": result.split_indices[1]},
        "boundary_scores": {"low_mid": result.boundary_scores[0], "mid_high": result.boundary_scores[1]},
        "balance_variance": result.balance_variance,
        "separation": result.separation,
        "objective": result.objective,
        "groups": {key: group_stats(result.groups[key]) for key in GROUP_KEYS},
    }


def save_recommended_groups(result: GroupingResult, output_path: Path) -> None:
    groups_payload = {}
    for key in GROUP_KEYS:
        stats = group_stats(result.groups[key])
        groups_payload[key] = {
            "name": {"low": "低分组", "mid": "中分组", "high": "高分组"}[key],
            "count": stats["count"],
            "mean_score": stats["mean"],
            "min_score": stats["min"],
            "max_score": stats["max"],
            "range": (
                f"< {result.boundary_scores[0]:.3f}"
                if key == "low"
                else (
                    f">= {result.boundary_scores[1]:.3f}"
                    if key == "high"
                    else f"[{result.boundary_scores[0]:.3f}, {result.boundary_scores[1]:.3f})"
                )
            ),
            "results": result.groups[key],
        }

    payload = {
        "method": result.name,
        "split_indices": {"low_mid": result.split_indices[0], "mid_high": result.split_indices[1]},
        "boundary_scores": {"low_mid": result.boundary_scores[0], "mid_high": result.boundary_scores[1]},
        "balance_variance": result.balance_variance,
        "separation": result.separation,
        "objective": result.objective,
        "groups": groups_payload,
    }

    dump_json(payload, output_path)
    print(f"推荐分组结果已保存: {output_path}")


def plot_group_comparison(
    scores: Sequence[float],
    strict: GroupingResult,
    quantile: GroupingResult,
    balanced: GroupingResult,
    output_path: Path,
) -> None:
    bins = min(30, max(10, len(scores) // 5))
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].hist(scores, bins=bins, alpha=0.7, color="#8e7cc3", edgecolor="black")
    axes[0, 0].set_title("Original Distribution")
    axes[0, 0].set_xlabel("Average Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    def plot_method(ax, result: GroupingResult) -> None:
        for key in GROUP_KEYS:
            values = [item["score"] for item in result.groups[key]]
            ax.hist(
                values,
                bins=bins,
                alpha=0.7,
                color=GROUP_COLORS[key],
                edgecolor="black",
                label=f"{key.capitalize()} (n={len(values)})",
            )
        ax.set_title(f"{result.name.capitalize()} Split")
        ax.set_xlabel("Average Score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plot_method(axes[0, 1], strict)
    plot_method(axes[1, 0], quantile)
    plot_method(axes[1, 1], balanced)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"分组对比图已生成: {output_path}")


def analyze_groupings(json_path: Path, output_dir: Path) -> None:
    _, results = load_data(json_path)
    records = sorted_records(results)
    scores = [item["score"] for item in records]

    strict_result = build_strict_grouping(records)
    quantile_result = build_quantile_grouping(records)
    balanced_result = build_balanced_grouping(records)

    summary = {
        "source_file": str(json_path.resolve()),
        "total_results": len(records),
        "methods": {
            "strict": summarize_method(strict_result),
            "quantile": summarize_method(quantile_result),
            "balanced": summarize_method(balanced_result),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "gpt_grouping_summary.json"
    dump_json(summary, summary_path)
    print(f"分组摘要已保存: {summary_path}")

    recommended_path = output_dir / "gpt_three_groups.json"
    save_recommended_groups(balanced_result, recommended_path)

    plot_path = output_dir / "gpt_grouping_comparison.png"
    plot_group_comparison(scores, strict_result, quantile_result, balanced_result, plot_path)

    print("\n=== 平衡方案统计 ===")
    for key in GROUP_KEYS:
        stats = group_stats(balanced_result.groups[key])
        print(
            f"{key.capitalize()}: 样本数 {stats['count']}, 平均分 {stats['mean']:.3f}, "
            f"范围 [{stats['min']:.3f}, {stats['max']:.3f}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对GPT评分进行分组分析。")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="输入的评分JSON文件路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="输出结果保存目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_groupings(args.input, args.output_dir)


if __name__ == "__main__":
    main()

