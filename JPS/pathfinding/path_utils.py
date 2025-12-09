from __future__ import annotations

from typing import Dict, List, Optional, Tuple


def normalize_direction(dx: int, dy: int) -> Tuple[int, int]:
    def sign(v: int) -> int:
        if v > 0:
            return 1
        if v < 0:
            return -1
        return 0

    return sign(dx), sign(dy)


def expand_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not path:
        return []

    expanded: List[Tuple[int, int]] = [path[0]]
    for a, b in zip(path, path[1:]):
        dx, dy = normalize_direction(b[0] - a[0], b[1] - a[1])
        cx, cy = a
        while (cx, cy) != b:
            cx += dx
            cy += dy
            expanded.append((cx, cy))
    return expanded


def reconstruct_path(
    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]], goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    node: Optional[Tuple[int, int]] = goal
    while node is not None:
        path.append(node)
        node = parent_map.get(node)
    path.reverse()
    return path
