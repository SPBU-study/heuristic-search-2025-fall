from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


def _deterministic_weight(ch: str) -> float:
    return 1.0 + (ord(ch) % 9)


def _is_obstacle(ch: str) -> bool:
    return ch in {"#", "@", "O", "T"}


def _load_weight_mapping(
        map_path: str, 
        terrain_weights_path: Optional[str]
) -> dict[str, float]:
    candidates: List[str] = []
    if terrain_weights_path:
        candidates.append(terrain_weights_path)
    else:
        map_dir = os.path.dirname(os.path.abspath(map_path))
        candidates.append(os.path.join(map_dir, "terrain_weights.json"))
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        candidates.append(os.path.join(repo_root, "terrain_weights.json"))

    for candidate in candidates:
        if not candidate:
            continue
        if not os.path.isfile(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Failed to load terrain weights from {candidate}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"Terrain weight file {candidate} must contain a JSON object")
        mapping: dict[str, float] = {}
        for key, value in data.items():
            if not isinstance(key, str) or len(key) == 0:
                continue
            try:
                mapping[key[0]] = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid weight for terrain '{key}' in {candidate}") from exc
        return mapping
    return {}


@dataclass
class WeightedGridMap:
    width: int
    height: int
    walkable: List[List[bool]]  # walkable[y][x]
    weights: List[List[float]]  # weights[y][x]
    chars: Optional[List[List[str]]] = None

    def __post_init__(self) -> None:
        if len(self.walkable) != self.height:
            raise ValueError("walkable rows do not match height")
        for row in self.walkable:
            if len(row) != self.width:
                raise ValueError("walkable columns do not match width")

        if len(self.weights) != self.height:
            raise ValueError("weights rows do not match height")
        for row in self.weights:
            if len(row) != self.width:
                raise ValueError("weights columns do not match width")

        if self.chars is None:
            self.chars = [["." if cell else "#" for cell in row] for row in self.walkable]

    @staticmethod
    def from_ascii(rows: List[str], weight_mapping: Optional[dict[str, float]] = None) -> "WeightedGridMap":
        if not rows:
            raise ValueError("ASCII rows cannot be empty")
        width = len(rows[0])
        height = len(rows)
        walkable: List[List[bool]] = []
        weights: List[List[float]] = []
        chars: List[List[str]] = []
        mapping = weight_mapping or {}

        for row in rows:
            if len(row) != width:
                raise ValueError("All rows must be the same length")
            row_walkable: List[bool] = []
            row_weights: List[float] = []
            row_chars = list(row)
            for ch in row_chars:
                if _is_obstacle(ch):
                    row_walkable.append(False)
                    row_weights.append(math.inf)
                else:
                    row_walkable.append(True)
                    weight = mapping.get(ch, _deterministic_weight(ch))
                    row_weights.append(float(weight))
            weights.append(row_weights)
            walkable.append(row_walkable)
            chars.append(row_chars)

        return WeightedGridMap(width=width, height=height, walkable=walkable, weights=weights, chars=chars)

    @staticmethod
    def from_movingai_map(path: str, terrain_weights_path: Optional[str] = None) -> "WeightedGridMap":
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = [line.rstrip("\n") for line in f if line.strip()]

        if len(raw_lines) < 4 or raw_lines[0].strip().lower() != "type octile":
            raise ValueError("Invalid map file: missing 'type octile' header")

        try:
            height = int(raw_lines[1].split()[1])
            width = int(raw_lines[2].split()[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("Invalid width/height declaration") from exc

        if raw_lines[3].strip().lower() != "map":
            raise ValueError("Missing 'map' line before grid data")

        weight_mapping = _load_weight_mapping(path, terrain_weights_path)

        grid_lines = raw_lines[4:]
        if len(grid_lines) < height:
            raise ValueError("Not enough rows for declared height")

        walkable: List[List[bool]] = []
        weights: List[List[float]] = []
        chars: List[List[str]] = []
        for row in grid_lines[:height]:
            if len(row) < width:
                raise ValueError("Row shorter than declared width")
            row_chars = list(row[:width])
            row_walkable: List[bool] = []
            row_weights: List[float] = []
            for ch in row_chars:
                if _is_obstacle(ch):
                    row_walkable.append(False)
                    row_weights.append(math.inf)
                else:
                    row_walkable.append(True)
                    weight = weight_mapping.get(ch, _deterministic_weight(ch))
                    row_weights.append(float(weight))
            walkable.append(row_walkable)
            weights.append(row_weights)
            chars.append(row_chars)

        return WeightedGridMap(width=width, height=height, walkable=walkable, weights=weights, chars=chars)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_walkable(self, x: int, y: int) -> bool:
        return self.in_bounds(x, y) and self.walkable[y][x]

    def valid_step(self, x: int, y: int, dx: int, dy: int) -> bool:
        nx, ny = x + dx, y + dy
        if not self.is_walkable(nx, ny):
            return False
        if dx != 0 and dy != 0:
            if not (self.is_walkable(x + dx, y) and self.is_walkable(x, y + dy)):
                return False
        return True

    def neighbors8(self, x: int, y: int) -> Iterable[Tuple[int, int]]:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                if self.valid_step(x, y, dx, dy):
                    yield (x + dx, y + dy)

    def min_cell_cost(self) -> float:
        best = math.inf
        for y in range(self.height):
            for x in range(self.width):
                if self.walkable[y][x]:
                    best = min(best, self.weights[y][x])
        return best

    def transition_cost(self, x: int, y: int, nx: int, ny: int) -> float:
        if not self.valid_step(x, y, nx - x, ny - y):
            raise ValueError("Invalid transition requested")

        wx, wy = self.weights[y][x], self.weights[ny][nx]
        if x == nx or y == ny:
            return (wx + wy) / 2.0

        u = (nx, y)
        v = (x, ny)
        wu = self.weights[u[1]][u[0]]
        wv = self.weights[v[1]][v[0]]
        return math.sqrt(2.0) * (wx + wu + wv + wy) / 4.0
