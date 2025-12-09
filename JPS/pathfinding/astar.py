from __future__ import annotations

import math
import heapq
from typing import Dict, List, Optional, Set, Tuple

from .grid import GridMap
from .heuristics import octile_distance, step_cost
from .path_utils import reconstruct_path


def astar_search(
    grid: GridMap, start: Tuple[int, int], goal: Tuple[int, int]
) -> Tuple[List[Tuple[int, int]], float, int]:
    open_heap: List[Tuple[float, int, int, int]] = []
    g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    closed: Set[Tuple[int, int]] = set()
    counter = 0
    expanded = 0

    heapq.heappush(open_heap, (octile_distance(start, goal), counter, start[0], start[1]))

    while open_heap:
        f, _, x, y = heapq.heappop(open_heap)
        node = (x, y)

        g_current = g_scores.get(node)
        
        if g_current is None:
            continue

        closed.add(node)
        expanded += 1

        if node == goal:
            return reconstruct_path(parent_map, goal), g_current, expanded

        for nx, ny in grid.neighbors8(x, y):
            neighbor = (nx, ny)
            if neighbor in closed:
                continue
            tentative_g = g_current + step_cost(nx - x, ny - y)
            if tentative_g + 1e-9 < g_scores.get(neighbor, math.inf):
                g_scores[neighbor] = tentative_g
                parent_map[neighbor] = node
                counter += 1
                heapq.heappush(
                    open_heap,
                    (tentative_g + octile_distance(neighbor, goal), counter, nx, ny),
                )

    return [], math.inf, expanded
