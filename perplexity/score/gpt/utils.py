#!/usr/bin/env python3
"""
GPT评分数据分析的通用工具函数。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "unformat-gpt-llm_score.json"
DEFAULT_DIMENSIONS: Tuple[str, ...] = (
    "generality",
    "relevance",
    "conciseness",
    "actionability",
    "average",
)
DEFAULT_INTERVAL_EDGES: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)


def load_data(json_path: Optional[Path] = None) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    """
    加载JSON文件并返回原始数据与results字典。
    如果顶层没有results字段，则默认整个字典即为结果集合。
    """
    path = Path(json_path) if json_path else DEFAULT_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(f"未找到JSON文件: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, Mapping) and "results" in data:
        results = data["results"]
    else:
        results = data

    if not isinstance(results, Mapping):
        raise ValueError("JSON文件的结构不符合预期，无法提取results数据。")

    return data, results


def collect_score_lists(
    results: Mapping[str, Mapping[str, Any]],
    dimensions: Sequence[str] = DEFAULT_DIMENSIONS,
) -> Dict[str, List[float]]:
    """
    收集指定维度的分数列表，返回维度到分数列表的映射。
    """
    scores_by_dim: Dict[str, List[float]] = {dim: [] for dim in dimensions}

    for item in results.values():
        scores = item.get("scores") or {}

        for dim in dimensions:
            entry = scores.get(dim)
            if dim == "average":
                entry = scores.get("average", entry)

            value = None
            if isinstance(entry, Mapping):
                value = entry.get("score")
            elif isinstance(entry, (int, float)):
                value = entry

            if value is None:
                continue

            try:
                scores_by_dim[dim].append(float(value))
            except (TypeError, ValueError):
                continue

    return scores_by_dim


def collect_records(
    results: Mapping[str, Mapping[str, Any]],
    extra_fields: Sequence[str] = ("question", "answer", "category"),
) -> List[Dict[str, Any]]:
    """
    将results转换为记录列表，默认包含id、score以及额外字段。
    """
    records: List[Dict[str, Any]] = []
    for result_id, item in results.items():
        scores = item.get("scores") or {}
        average = scores.get("average")
        if isinstance(average, Mapping):
            score = average.get("score")
        else:
            score = average

        if score is None:
            continue

        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue

        record: Dict[str, Any] = {"id": result_id, "score": score_value}

        for field in extra_fields:
            if field in item:
                record[field] = item[field]

        records.append(record)

    return records


def compute_basic_statistics(values: Sequence[float]) -> Dict[str, Any]:
    """
    计算基础统计量，返回包含count、mean、median等信息的字典。
    """
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }

    array = np.asarray(values, dtype=float)

    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=0)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def compute_interval_distribution(
    values: Sequence[float],
    edges: Sequence[float] = DEFAULT_INTERVAL_EDGES,
) -> Dict[str, int]:
    """
    根据给定边界计算分数区间的分布。
    """
    if not values:
        return {}

    counts: Dict[str, int] = {}
    array = np.asarray(values, dtype=float)

    for start, end in zip(edges[:-1], edges[1:]):
        label = f"{start:.0f}-{end:.0f}"
        mask = (array >= start) & (array < end)
        counts[label] = int(np.count_nonzero(mask))

    last_edge = edges[-1]
    label = f">={last_edge:.0f}"
    counts[label] = int(np.count_nonzero(array >= last_edge))

    return counts


def dump_json(data: Mapping[str, Any], output_path: Path) -> None:
    """
    将数据写入JSON文件（UTF-8，带缩进）。
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


__all__ = [
    "DEFAULT_DATA_PATH",
    "DEFAULT_DIMENSIONS",
    "DEFAULT_INTERVAL_EDGES",
    "collect_records",
    "collect_score_lists",
    "compute_basic_statistics",
    "compute_interval_distribution",
    "dump_json",
    "load_data",
]

