#!/usr/bin/env python3
"""
Visualize a single step by overlaying the chosen frontier region and the agent
position on the saved top-down voxel map. All other icons are intentionally
omitted to keep the figure minimal.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

_BASE_MAP_COLORS = np.asarray(
    [
        [255, 255, 255],  # background
        [200, 200, 200],  # free space
        [194, 246, 198],  # explored
        [100, 100, 100],  # nearby obstacles
        [0, 0, 0],  # solid obstacles
    ],
    dtype=np.float32,
)


def _load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_float_image(img: np.ndarray) -> np.ndarray:
    arr = img.astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def _circle_mask(
    shape: Sequence[int], center: Optional[Sequence[float]], radius: float
) -> Optional[np.ndarray]:
    if center is None:
        return None
    cy, cx = center
    rr, cc = np.ogrid[: shape[0], : shape[1]]
    mask = (rr - cy) ** 2 + (cc - cx) ** 2 <= radius**2
    return mask


def _apply_overlay(
    image: np.ndarray, mask: np.ndarray, color: Sequence[float], alpha: float
) -> np.ndarray:
    if mask is None:
        return image
    overlay = image.copy()
    rgb_slice = slice(0, 3)
    overlay_region = overlay[..., rgb_slice]
    color = np.asarray(color, dtype=np.float32)
    overlay_region[mask] = (
        overlay_region[mask] * (1 - alpha) + color * alpha
    )
    return overlay


def plot_frontier_step(
    run_dir: Path,
    env_id: str,
    question_id: str,
    step_idx: int,
    output_path: Path,
    frontier_radius: float,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    question_dir = (run_dir / question_id).resolve()
    if not question_dir.exists():
        raise FileNotFoundError(f"Question directory not found: {question_dir}")

    map_path = question_dir / "visualization" / f"{step_idx}_map.png"
    if not map_path.exists():
        raise FileNotFoundError(
            f"Top-down map not found for step {step_idx}: {map_path}"
        )

    replay = _load_json(run_dir / "replay_step_info.json")
    if env_id not in replay:
        raise KeyError(f"Env id '{env_id}' missing in replay_step_info.json")
    if question_id not in replay[env_id]:
        raise KeyError(
            f"Question id '{question_id}' not found under env '{env_id}'"
        )

    step_key = f"step_{step_idx}"
    coords = _load_json(run_dir / "coords_all.json")
    question_coords = coords.get(question_id, {})
    coords_step = question_coords.get(step_key, {})
    agent_voxel = coords_step.get("agent_position_voxel")
    target_voxel = coords_step.get("target_position")

    map_img = mpimg.imread(map_path)
    map_img = _ensure_float_image(map_img)

    palette = _BASE_MAP_COLORS / 255.0
    flat = map_img[..., :3].reshape(-1, 3)
    dist = np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=-1)
    palette_idx = np.argmin(dist, axis=1).reshape(map_img.shape[0], map_img.shape[1])
    palette_idx = ndimage.median_filter(palette_idx, size=7)
    map_img[..., :3] = palette[palette_idx]

    frontier_mask = _circle_mask(map_img.shape[:2], target_voxel, frontier_radius)
    frontier_color = np.array([249, 142, 63]) / 255.0
    map_img = _apply_overlay(map_img, frontier_mask, frontier_color, alpha=0.55)

    h, w = map_img.shape[:2]
    fig_height = 8 * (h / w)
    fig, ax = plt.subplots(figsize=(8, max(6, fig_height)))
    ax.imshow(map_img)
    ax.set_xticks([])
    ax.set_yticks([])

    if frontier_mask is not None and target_voxel:
        circle = plt.Circle(
            (target_voxel[1], target_voxel[0]),
            radius=frontier_radius,
            facecolor="none",
            edgecolor="#ff4f5e",
            linewidth=1.5,
        )
        ax.add_patch(circle)

    if agent_voxel:
        ax.scatter(
            agent_voxel[1],
            agent_voxel[0],
            color="#17bcf3",
            s=70,
            edgecolors="white",
            linewidths=1.2,
            zorder=5,
        )

    question_text = replay[env_id][question_id].get("question", "").strip()
    subtitle = f"{question_id} • step {step_idx}"
    if question_text:
        subtitle = f"{question_text}  ({subtitle})"
    ax.set_title(subtitle, fontsize=12)

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize the chosen frontier region for a single step."
    )
    parser.add_argument("--run_dir", required=True, help="Root directory of the run")
    parser.add_argument("--env_id", required=True, help="Episode/environment id key")
    parser.add_argument("--question_id", required=True, help="Question UUID")
    parser.add_argument("--step", type=int, required=True, help="Step index to plot")
    parser.add_argument(
        "--output",
        type=str,
        help="Output PNG path (default: teaser/<question>_step<step>_frontier.png)",
    )
    parser.add_argument(
        "--frontier_radius",
        type=float,
        default=4.0,
        help="Radius (in voxels) for shading the frontier region.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    if args.output:
        output_path = Path(args.output)
    else:
        default_dir = Path("teaser")
        output_path = default_dir / f"{args.question_id}_step{args.step}_frontier.png"

    saved_path = plot_frontier_step(
        run_dir=run_dir,
        env_id=args.env_id,
        question_id=args.question_id,
        step_idx=args.step,
        output_path=output_path,
        frontier_radius=args.frontier_radius,
    )
    print(f"Saved visualization to {saved_path}")


if __name__ == "__main__":
    main()

