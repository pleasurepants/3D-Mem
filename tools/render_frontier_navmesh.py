#!/usr/bin/env python3
"""
Render a frontier visualization directly from the Habitat navmesh (voxel map),
without using the already-rendered `visualization/*.png` outputs.

Given a scene id, replay coordinates, and a step index, this script:
  1. Loads the HM3D navmesh for the scene via Habitat PathFinder.
  2. Samples a top-down occupancy grid at voxel_size resolution.
  3. Overlays the agent trajectory (up to the queried step) and the current
     frontier target converted from voxel indices to world coordinates.

The result is a new PNG that reflects the raw voxel/navigability map.
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_coords(coords_path: Path, question_id: str) -> Dict[str, Dict]:
    with coords_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if question_id not in data:
        raise KeyError(f"Question id '{question_id}' not found in {coords_path}")
    return data[question_id]


def _get_scene_navmesh(scene_id: str, scene_data_path: Path) -> Tuple["habitat_sim.PathFinder", np.ndarray, np.ndarray]:
    try:
        import habitat_sim
    except ImportError as exc:
        raise RuntimeError(
            "habitat_sim is required. Please activate the 3dmem environment first."
        ) from exc

    split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
    navmesh_path = (
        scene_data_path / split / scene_id / f"{scene_id.split('-')[1]}.basis.navmesh"
    )
    if not navmesh_path.exists():
        raise FileNotFoundError(f"Navmesh not found: {navmesh_path}")

    pf = habitat_sim.PathFinder()
    pf.load_nav_mesh(navmesh_path.as_posix())
    min_bound, max_bound = pf.get_bounds()
    return pf, np.array(min_bound, dtype=np.float32), np.array(max_bound, dtype=np.float32)


def _build_voxel_grid(
    pf: "habitat_sim.PathFinder",
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    voxel_size: float,
    snap_thresh: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    min_x, min_y, min_z = min_bound
    max_x, max_y, max_z = max_bound

    n_x = int(math.ceil((max_x - min_x) / voxel_size)) + 1
    n_z = int(math.ceil((max_z - min_z) / voxel_size)) + 1
    grid = np.zeros((n_z, n_x), dtype=np.uint8)

    for iz in range(n_z):
        z = max_z - iz * voxel_size
        for ix in range(n_x):
            x = min_x + ix * voxel_size
            point = np.array([x, min_y + 0.2, z], dtype=np.float32)
            nav = pf.is_navigable(point)
            if not nav:
                snapped = np.array(pf.snap_point(point), dtype=np.float32)
                if np.linalg.norm(snapped - point) < snap_thresh:
                    nav = True
            grid[iz, ix] = 1 if nav else 0

    xs = min_x + np.arange(n_x) * voxel_size
    zs = max_z - np.arange(n_z) * voxel_size
    return grid, xs, zs


def _voxel_to_world(
    voxel_idx: Sequence[int],
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    # voxel_idx is [row(x-axis), col(z-axis)] in planner coordinates
    vx, vz = voxel_idx
    world_x = min_bound[0] + voxel_size * vx
    world_z = max_bound[2] - voxel_size * vz
    return np.array([world_x, world_z], dtype=np.float32)


def _world_to_grid(world: Sequence[float], min_bound: np.ndarray, max_bound: np.ndarray, voxel_size: float) -> Tuple[int, int]:
    x, z = world
    ix = int(round((x - min_bound[0]) / voxel_size))
    iz = int(round((max_bound[2] - z) / voxel_size))
    return ix, iz


def _colorize_grid(grid: np.ndarray) -> np.ndarray:
    color = np.zeros((*grid.shape, 3), dtype=np.float32)
    color[:, :] = np.array([0.1, 0.1, 0.1], dtype=np.float32)
    color[grid == 1] = np.array([0.82, 0.82, 0.82], dtype=np.float32)
    return color


def render_frontier_from_navmesh(
    scene_id: str,
    coords_path: Path,
    question_id: str,
    step_idx: int,
    scene_data_path: Path,
    output_path: Path,
    voxel_size: float,
    frontier_radius_vox: float,
) -> Path:
    coords = _load_coords(coords_path, question_id)
    step_key = f"step_{step_idx}"
    if step_key not in coords:
        raise KeyError(f"{step_key} missing for {question_id} in {coords_path}")

    pf, min_bound, max_bound = _get_scene_navmesh(scene_id, scene_data_path)
    grid, xs, zs = _build_voxel_grid(pf, min_bound, max_bound, voxel_size)
    color_map = _colorize_grid(grid)

    # Trajectory up to step
    path_world = [np.array(coords[f"step_{i}"]["agent_position"])[[0, 2]] for i in range(step_idx + 1)]
    path_idxs = np.array(
        [_world_to_grid(p, min_bound, max_bound, voxel_size) for p in path_world],
        dtype=np.int32,
    )

    current_voxel = coords[step_key]["agent_position_voxel"]
    current_world = _voxel_to_world(current_voxel, min_bound, max_bound, voxel_size)
    current_idx = _world_to_grid(current_world, min_bound, max_bound, voxel_size)

    target_voxel = coords[step_key]["target_position"]
    target_world = _voxel_to_world(target_voxel, min_bound, max_bound, voxel_size)
    target_idx = _world_to_grid(target_world, min_bound, max_bound, voxel_size)

    fig_h = 8 * (grid.shape[0] / grid.shape[1])
    fig, ax = plt.subplots(figsize=(8, max(6, fig_h)))
    ax.imshow(color_map, origin="upper")
    ax.axis("off")

    ax.plot(
        path_idxs[:, 0],
        path_idxs[:, 1],
        color="#ffffff",
        linewidth=1.5,
        alpha=0.9,
    )
    ax.scatter(
        path_idxs[0, 0],
        path_idxs[0, 1],
        color="#86d4ff",
        s=60,
        edgecolors="black",
        linewidths=0.5,
        label="Start",
    )
    ax.scatter(
        current_idx[0],
        current_idx[1],
        color="#17bcf3",
        s=60,
        edgecolors="white",
        linewidths=1.0,
        label="Agent",
    )
    ax.scatter(
        target_idx[0],
        target_idx[1],
        color="#ff4f5e",
        s=70,
        edgecolors="white",
        linewidths=1.0,
        label="Frontier center",
    )

    circ = plt.Circle(
        (target_idx[0], target_idx[1]),
        radius=frontier_radius_vox,
        facecolor="none",
        edgecolor="#ff4f5e",
        linewidth=1.3,
    )
    ax.add_patch(circ)

    ax.set_title(f"{scene_id} • {question_id} • step {step_idx}", fontsize=11)
    ax.legend(loc="upper right", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render frontier visualization from navmesh/voxel map."
    )
    parser.add_argument("--scene_id", required=True, help="HM3D scene id (e.g. 00824-Dd4bFSTQ8gi)")
    parser.add_argument("--coords_json", required=True, help="Path to coords_all.json")
    parser.add_argument("--question_id", required=True, help="Question UUID")
    parser.add_argument("--step", type=int, required=True, help="Step index")
    parser.add_argument(
        "--scene_data_path",
        default="/anvme/workspace/v100dd12-3dmem/hm3d/data/3dmem",
        help="Root directory containing HM3D scene assets",
    )
    parser.add_argument(
        "--vox_size",
        type=float,
        default=0.1,
        help="Voxel size used by TSDF planner (meters per cell)",
    )
    parser.add_argument(
        "--frontier_radius",
        type=float,
        default=4.0,
        help="Radius (in voxel units) for highlighting the frontier region",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Destination PNG path",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = render_frontier_from_navmesh(
        scene_id=args.scene_id,
        coords_path=Path(args.coords_json),
        question_id=args.question_id,
        step_idx=args.step,
        scene_data_path=Path(args.scene_data_path),
        output_path=Path(args.output),
        voxel_size=args.vox_size,
        frontier_radius_vox=args.frontier_radius,
    )
    print(f"Saved navmesh visualization to {output_path}")


if __name__ == "__main__":
    main()

