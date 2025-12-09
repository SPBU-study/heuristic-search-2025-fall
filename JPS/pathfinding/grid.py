from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass
class GridMap:
    width: int
    height: int
    walkable: List[List[bool]] # walkable[y][x]
    chars: Optional[List[List[str]]] = None # chars[y][x]

    def __post_init__(self) -> None:
        if len(self.walkable) != self.height:
            raise ValueError("walkable rows do not match height")
        for row in self.walkable:
            if len(row) != self.width:
                raise ValueError("walkable columns do not match width")
        if self.chars is None:
            self.chars = [["." if cell else "#" for cell in row] for row in self.walkable]

    @staticmethod
    def from_movingai_map(path: str) -> "GridMap":
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

        grid_lines = raw_lines[4:]
        if len(grid_lines) < height:
            raise ValueError("Not enough rows for declared height")

        walkable: List[List[bool]] = []
        chars: List[List[str]] = []
        for row in grid_lines[:height]:
            if len(row) < width:
                raise ValueError("Row shorter than declared width")
            row_chars = list(row[:width])
            row_walkable: List[bool] = []
            for ch in row_chars:
                if ch in {".", "G", "S", "W"}:
                    row_walkable.append(True)
                elif ch in {"@", "O", "T"}:
                    row_walkable.append(False)
                else:
                    row_walkable.append(False)
            walkable.append(row_walkable)
            chars.append(row_chars)

        return GridMap(width=width, height=height, walkable=walkable, chars=chars)

    @staticmethod
    def from_ascii(rows: List[str]) -> "GridMap":
        if not rows:
            raise ValueError("ASCII rows cannot be empty")
        width = len(rows[0])
        height = len(rows)
        walkable: List[List[bool]] = []
        chars: List[List[str]] = []
        for row in rows:
            if len(row) != width:
                raise ValueError("All rows must be the same length")
            row_chars = list(row)
            row_walkable = [ch != "#" for ch in row_chars]
            walkable.append(row_walkable)
            chars.append(row_chars)
        return GridMap(width=width, height=height, walkable=walkable, chars=chars)

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


@dataclass
class ScenarioProblem:
    map_path: str
    map_width: int
    map_height: int
    start_x: int
    start_y: int
    goal_x: int
    goal_y: int
    optimal_length: float


def load_scenarios(path: str) -> List[ScenarioProblem]:
    problems: List[ScenarioProblem] = []

    scen_dir = os.path.dirname(os.path.abspath(path))
    scen_dir_name = os.path.basename(scen_dir)
    parent_dir = os.path.dirname(scen_dir)

    if scen_dir_name.endswith("-scen"):
        map_dir_name = scen_dir_name[:-5] + "-map"
        base_dir = os.path.join(parent_dir, map_dir_name)
    else:
        base_dir = scen_dir

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.lower().startswith("version"):
                continue
            parts = stripped.split()
            if len(parts) < 8:
                continue
            try:
                if len(parts) >= 9:
                    _, map_rel, w, h, sx, sy, gx, gy, optimal = parts[:9]
                else:
                    map_rel, w, h, sx, sy, gx, gy, optimal = parts
                problems.append(
                    ScenarioProblem(
                        map_path=os.path.join(base_dir, map_rel),
                        map_width=int(w),
                        map_height=int(h),
                        start_x=int(sx),
                        start_y=int(sy),
                        goal_x=int(gx),
                        goal_y=int(gy),
                        optimal_length=float(optimal),
                    )
                )
            except ValueError:
                continue
    return problems
