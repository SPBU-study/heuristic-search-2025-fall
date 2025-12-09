from __future__ import annotations

import os
import string
import time
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import Circle

from .grid import GridMap
from .path_utils import expand_path
from .weighted_grid import WeightedGridMap

GridLike = Union[GridMap, WeightedGridMap]


def _build_unweighted_image(grid: GridMap) -> np.ndarray:
    h, w = grid.height, grid.width
    img = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            img[y, x] = 1 if grid.walkable[y][x] else 0
    return img


def _build_weighted_terrain_image(grid: WeightedGridMap) -> Tuple[np.ndarray, List]:
    h, w = grid.height, grid.width
    chars = grid.chars

    terrain_palette: List[str] = [
        "#cddafd",
        "#d0f4de",
        "#f4d7da",
        "#d9e4dd",
        "#fce2ce",
        "#e5d4ef",
        "#c5e1f5",
        "#f0d3f7",
        "#d3f0f7",
        "#f7d6bf",
        "#d4e6a5",
        "#f0c0c0",
        "#c0d8f0",
        "#e6c8dc",
        "#c3eddc",
        "#f3d1c2",
        "#d2d1f0",
        "#c6e0b4",
        "#e8c6c0",
        "#c2d6e6",
        "#e1c8f0",
        "#c8e9f3",
        "#f2c6da",
        "#dce3c4",
        "#c4d4f2",
        "#e7d0c8",
    ]

    def terrain_color(ch: str) -> str:
        key = ch.lower()
        if key and key in string.ascii_lowercase:
            idx = string.ascii_lowercase.index(key)
        elif key:
            idx = ord(key[0]) % len(terrain_palette)
        else:
            idx = 0
        return terrain_palette[idx % len(terrain_palette)]

    walkable_chars = sorted(
        {chars[y][x] for y in range(h) for x in range(w) if grid.walkable[y][x]}
    )

    terrain_to_idx: Dict[str, int] = {}
    colors: List[str] = ["#000000"]
    for ch in walkable_chars:
        terrain_to_idx[ch] = len(colors)
        colors.append(terrain_color(ch))

    img = np.zeros((h, w), dtype=int)
    for y in range(h):
        for x in range(w):
            if not grid.walkable[y][x]:
                img[y, x] = 0
            else:
                ch = chars[y][x]
                img[y, x] = terrain_to_idx.get(ch, len(colors))
                if img[y, x] == len(colors):
                    colors.append(terrain_color(ch))
    return img, colors


def _cell_size_points(ax: plt.Axes, grid: GridLike) -> float:
    fig = ax.figure
    fig.canvas.draw()

    bbox = ax.get_window_extent()
    cell_w_px = bbox.width / grid.width
    cell_h_px = bbox.height / grid.height
    cell_px = min(cell_w_px, cell_h_px)

    points_per_pixel = 72.0 / fig.dpi
    return cell_px * points_per_pixel


def _patch_linewidth(cell_points: float, base: float, *, min_w: float, max_w: float) -> float:
    return max(min_w, min(max_w, cell_points * base))


def render_grid_path(
    grid: GridLike,
    path: List[Tuple[int, int]],
    *,
    jump_points: Optional[List[Tuple[int, int]]] = None,
    start: Optional[Tuple[int, int]] = None,
    goal: Optional[Tuple[int, int]] = None,
    output_path: Optional[str] = None,
    title: str = "",
    show: bool = False,
) -> str:
    path_for_plot = path or []
    if start is None and path_for_plot:
        start = path_for_plot[0]
    if goal is None and path_for_plot:
        goal = path_for_plot[-1]

    max_side = max(grid.width, grid.height)
    base_size = 6.0
    scale = max(1.0, max_side / 256.0)
    figsize = (base_size * scale, base_size * scale)

    fig, ax = plt.subplots(figsize=figsize, dpi=500)

    if isinstance(grid, WeightedGridMap):
        img, colors = _build_weighted_terrain_image(grid)
        cmap = plt.matplotlib.colors.ListedColormap(colors)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(len(colors) + 1) - 0.5, ncolors=len(colors))
    else:
        img = _build_unweighted_image(grid)
        cmap = plt.matplotlib.colors.ListedColormap(["#000000", "#e6e6e6"])
        norm = mcolors.NoNorm()

    ax.imshow(
        img,
        origin="upper",
        extent=(0, grid.width, grid.height, 0),
        interpolation="none",
        cmap=cmap,
        norm=norm,
        zorder=0,
    )

    ax.set_aspect("equal")
    ax.set_xlim(0, grid.width)
    ax.set_ylim(grid.height, 0)

    ax.set_xticks(np.arange(0, grid.width + 1), minor=True)
    ax.set_yticks(np.arange(0, grid.height + 1), minor=True)
    
    ax.grid(which="minor", color="#b0b0b0", linewidth=0.4, zorder=1)

    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    cell_points = _cell_size_points(ax, grid)
    circle_radius = 0.13

    outline_width = _patch_linewidth(cell_points, 0.16, min_w=0.8, max_w=1.5)

    path_width = _patch_linewidth(cell_points, 0.1, min_w=0.3, max_w=1.5)

    def add_circle(point: Tuple[int, int], face: str, edge: str, z: int) -> None:
        cx, cy = point
        marker = Circle(
            (cx + 0.5, cy + 0.5),
            radius=circle_radius,
            facecolor=face,
            edgecolor=edge,
            linewidth=outline_width,
            zorder=z,
        )
        ax.add_patch(marker)

    if path_for_plot:
        xs = [x + 0.5 for x, _ in path_for_plot]
        ys = [y + 0.5 for _, y in path_for_plot]
        ax.plot(xs, ys, color="black", linewidth=path_width, zorder=2)

    if jump_points:
        for jp in jump_points:
            add_circle(jp, "#ffd700", "black", 3)

    if start is not None:
        add_circle(start, "#00bfff", "black", 4)
    if goal is not None:
        add_circle(goal, "#ff1493", "black", 4)

    if title:
        ax.set_title(title)

    if output_path is None:
        root = os.path.dirname(os.path.dirname(__file__))
        out_dir = os.path.join(root, "assets")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        fname = f"{ts}_grid_{grid.width}x{grid.height}.png"
        output_path = os.path.join(out_dir, fname)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    if not show:
        plt.close(fig)

    return os.path.abspath(output_path)
