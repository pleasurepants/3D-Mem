#!/usr/bin/env python3
"""
针对GPT评分数据的统计与可视化分析。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from utils import (
    DEFAULT_DATA_PATH,
    collect_score_lists,
    compute_basic_statistics,
    compute_interval_distribution,
    dump_json,
    load_data,
)

OUTPUT_DIR = Path(__file__).resolve().parent


def create_average_distribution_plot(scores: Sequence[float], output_path: Path) -> None:
    """绘制平均分的直方图、箱线图、密度图与CDF。"""
    if not scores:
        print("警告: 平均分列表为空，跳过平均分分布图生成。")
        return

    array = np.asarray(scores, dtype=float)
    bins = min(30, max(10, len(array) // 5))

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 直方图
    axes[0, 0].hist(array, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Average Score Histogram")
    axes[0, 0].set_xlabel("Average Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    # 箱线图
    axes[0, 1].boxplot(array, vert=True)
    axes[0, 1].set_title("Average Score Box Plot")
    axes[0, 1].set_ylabel("Average Score")
    axes[0, 1].grid(True, alpha=0.3)

    # 密度图
    axes[1, 0].hist(array, bins=bins, density=True, alpha=0.7, color="lightgreen", edgecolor="black")
    axes[1, 0].set_title("Average Score Density")
    axes[1, 0].set_xlabel("Average Score")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].grid(True, alpha=0.3)

    # 累积分布函数
    sorted_scores = np.sort(array)
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
    axes[1, 1].plot(sorted_scores, cumulative, marker="o", markersize=2, linewidth=2)
    axes[1, 1].set_title("Average Score CDF")
    axes[1, 1].set_xlabel("Average Score")
    axes[1, 1].set_ylabel("Cumulative Probability")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"平均分分布图已生成: {output_path}")


def create_dimension_histograms(score_map: Dict[str, Sequence[float]], output_path: Path) -> None:
    """绘制各维度分数的直方图。"""
    dimensions = [dim for dim in score_map.keys() if dim != "average"]
    available_dims = [dim for dim in dimensions if score_map.get(dim)]

    if not available_dims:
        print("警告: 未找到维度分数数据，跳过维度直方图生成。")
        return

    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes_flat = axes.flatten()

    for ax, dim in zip(axes_flat, available_dims):
        values = np.asarray(score_map[dim], dtype=float)
        bins = np.arange(0.5, 5.5, 1.0)
        ax.hist(values, bins=bins, alpha=0.75, color="#6fa8dc", edgecolor="black")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_ylim(bottom=0)
        ax.set_title(f"{dim.capitalize()} Score Distribution")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    # 隐藏多余子图
    for ax in axes_flat[len(available_dims) :]:
        ax.set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"维度分布图已生成: {output_path}")


def format_stat_line(label: str, value: Any) -> str:
    if value is None:
        return f"{label}: N/A"
    if isinstance(value, float):
        return f"{label}: {value:.3f}"
    return f"{label}: {value}"


def analyze_scores(json_path: Path, output_dir: Path) -> Dict[str, Any]:
    """执行完整分析流程并返回统计摘要。"""
    _, results = load_data(json_path)
    score_lists = collect_score_lists(results)
    average_scores = score_lists.get("average", [])

    stats_by_dim = {dim: compute_basic_statistics(values) for dim, values in score_lists.items()}

    summary: Dict[str, Any] = {
        "source_file": str(json_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "total_results": len(results),
        "statistics": stats_by_dim,
        "average_interval_distribution": compute_interval_distribution(average_scores),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "gpt_score_summary.json"
    dump_json(summary, summary_path)
    print(f"统计摘要已保存: {summary_path}")

    average_plot_path = output_dir / "gpt_score_distribution.png"
    create_average_distribution_plot(average_scores, average_plot_path)

    dimension_plot_path = output_dir / "gpt_dimension_distribution.png"
    create_dimension_histograms(score_lists, dimension_plot_path)

    avg_stats = stats_by_dim.get("average", {})
    print("\n=== 平均分统计 ===")
    for key in ("count", "mean", "median", "std", "min", "max", "q25", "q75"):
        print(format_stat_line(key, avg_stats.get(key)))

    print("\n=== 各维度有效样本数 ===")
    for dim, values in score_lists.items():
        print(f"{dim}: {len(values)}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析GPT评分结果并生成可视化。")
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
        help="保存可视化与统计结果的目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_scores(args.input, args.output_dir)


if __name__ == "__main__":
    main()

