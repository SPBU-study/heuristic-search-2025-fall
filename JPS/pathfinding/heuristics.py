from __future__ import annotations

import math
from typing import Tuple

DIAGONAL_DISTANCE: float = math.sqrt(2.0)


def step_cost(dx: int, dy: int) -> float:
    return 1.0 if dx == 0 or dy == 0 else DIAGONAL_DISTANCE


def octile_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    D = 1.0
    D2 = DIAGONAL_DISTANCE
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
