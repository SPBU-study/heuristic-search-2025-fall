from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Set, Tuple

from .heuristics import weighted_octile_distance
from .path_utils import reconstruct_path
from .weighted_grid import WeightedGridMap


def astarw_search(
    grid: WeightedGridMap, start: Tuple[int, int], goal: Tuple[int, int]
) -> Tuple[List[Tuple[int, int]], float, int, float]:
    start_time = time.perf_counter()
    open_heap: List[Tuple[float, int, int, int]] = []
    g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    closed: Set[Tuple[int, int]] = set()
    counter = 0
    expanded = 0

    heapq.heappush(open_heap, (weighted_octile_distance(start, goal, grid), counter, start[0], start[1]))

    while open_heap:
        f, _, x, y = heapq.heappop(open_heap)
        node = (x, y)
        if node in closed:
            continue

        g_current = g_scores.get(node)
        if g_current is None:
            continue

        if f > g_current + weighted_octile_distance(node, goal, grid) + 1e-9:
            continue

        closed.add(node)
        expanded += 1

        if node == goal:
            path = reconstruct_path(parent_map, goal)
            elapsed_time = time.perf_counter() - start_time
            return path, g_current, expanded, elapsed_time

        for nx, ny in grid.neighbors8(x, y):
            neighbor = (nx, ny)
            if neighbor in closed:
                continue
            tentative_g = g_current + grid.transition_cost(x, y, nx, ny)
            if tentative_g + 1e-9 < g_scores.get(neighbor, math.inf):
                g_scores[neighbor] = tentative_g
                parent_map[neighbor] = node
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative_g + weighted_octile_distance(neighbor, goal, grid), counter, nx, ny),
                )

    elapsed_time = time.perf_counter() - start_time
    return [], math.inf, expanded, elapsed_time
