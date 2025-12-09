from __future__ import annotations

import heapq
import math
import time
from typing import Dict, List, Optional, Set, Tuple

from .grid import GridMap
from .heuristics import DIAGONAL_DISTANCE, octile_distance
from .path_utils import reconstruct_path

DIRECTIONS_8: List[Tuple[int, int]] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),    
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]


def prune_neighbors(
    grid: GridMap, 
    current: Tuple[int, int], 
    parent: Optional[Tuple[int, int]]
) -> List[Tuple[int, int]]:

    x, y = current
    if parent is None:
        return [(dx, dy) for dx, dy in DIRECTIONS_8 if grid.valid_step(x, y, dx, dy)]

    px, py = parent
    dx, dy = x - px, y - py
    pruned: List[Tuple[int, int]] = []

    if dx != 0 and dy != 0: # Diagonal movement
        for ndx, ndy in ((dx, dy), (dx, 0), (0, dy)):
            if grid.valid_step(x, y, ndx, ndy):
                pruned.append((ndx, ndy))
    else: 
        if dx != 0: # Horizontal movement.
            if grid.valid_step(x, y, dx, 0):
                pruned.append((dx, 0))
            for ty in (-1, 1):
                if not grid.valid_step(px, py, dx, ty):
                    if grid.valid_step(x, y, 0, ty):
                        pruned.append((0, ty))
                    if grid.valid_step(x, y, dx, ty):
                        pruned.append((dx, ty))
        else: # Vertical movement.
            if grid.valid_step(x, y, 0, dy):
                pruned.append((0, dy))
            for tx in (-1, 1):
                if not grid.valid_step(px, py, tx, dy):
                    if grid.valid_step(x, y, tx, 0):
                        pruned.append((tx, 0))
                    if grid.valid_step(x, y, tx, dy):
                        pruned.append((tx, dy))

    return list(dict.fromkeys(pruned))


def _has_forced_neighbor_straight(
    grid: GridMap,
    x: int,
    y: int,
    dx: int,
    dy: int,
) -> bool:

    px, py = x - dx, y - dy

    if dx != 0 and dy == 0:
        for ty in (-1, 1):
            if not grid.valid_step(px, py, dx, ty):
                if grid.valid_step(x, y, 0, ty) or grid.valid_step(x, y, dx, ty):
                    return True
        return False

    if dy != 0 and dx == 0:
        for tx in (-1, 1):
            if not grid.valid_step(px, py, tx, dy):
                if grid.valid_step(x, y, tx, 0) or grid.valid_step(x, y, tx, dy):
                    return True
        return False

def jump(
    grid: GridMap, 
    x: int, y: int, 
    dx: int, dy: int, 
    goal: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    if dx == 0 and dy == 0:
        return None

    while True:
        if not grid.valid_step(x, y, dx, dy):
            return None

        x += dx
        y += dy

        if (x, y) == goal:
            return (x, y)

        if dx == 0 or dy == 0:
            if _has_forced_neighbor_straight(grid, x, y, dx, dy):
                return (x, y)
        else:
            if jump(grid, x, y, dx, 0, goal) is not None:
                return (x, y)
            if jump(grid, x, y, 0, dy, goal) is not None:
                return (x, y)

def identify_successors(
    grid: GridMap,
    current: Tuple[int, int],
    prune_parent: Optional[Tuple[int, int]],
    goal: Tuple[int, int],
    g_scores: Dict[Tuple[int, int], float],
    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    dir_parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
) -> List[Tuple[int, int]]:

    x, y = current
    successors: List[Tuple[int, int]] = []
    directions = prune_neighbors(grid, current, prune_parent)

    for dx, dy in directions:
        jp = jump(grid, x, y, dx, dy, goal)
        if jp is None:
            continue

        jx, jy = jp
        steps = max(abs(jx - x), abs(jy - y))
        move_cost = float(steps) * (DIAGONAL_DISTANCE if dx != 0 and dy != 0 else 1.0)
        tentative_g = g_scores[current] + move_cost

        if tentative_g + 1e-9 < g_scores.get((jx, jy), math.inf):
            g_scores[(jx, jy)] = tentative_g

            parent_map[(jx, jy)] = current

            dir_parent[(jx, jy)] = (jx - dx, jy - dy)

            successors.append((jx, jy))

    return successors

def jump_point_search(
    grid: GridMap, start: Tuple[int, int], goal: Tuple[int, int]
) -> Tuple[List[Tuple[int, int]], float, int, float]:
    start_time = time.perf_counter()
    open_heap: List[Tuple[float, int, int, int]] = []
    g_scores: Dict[Tuple[int, int], float] = {start: 0.0}

    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    dir_parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

    closed: Set[Tuple[int, int]] = set()
    counter = 0
    expanded = 0

    heapq.heappush(open_heap, (octile_distance(start, goal), counter, start[0], start[1]))

    while open_heap:
        f, _, x, y = heapq.heappop(open_heap)
        node = (x, y)
        if node in closed:
            continue

        g_current = g_scores.get(node)
        if g_current is None:
            continue

        if f > g_current + octile_distance(node, goal) + 1e-9:
            continue

        closed.add(node)
        expanded += 1

        if node == goal:
            elapsed_time = time.perf_counter() - start_time
            return reconstruct_path(parent_map, goal), g_current, expanded, elapsed_time

        prune_parent = dir_parent.get(node)
        successors = identify_successors(
            grid,
            node,
            prune_parent,
            goal,
            g_scores,
            parent_map,
            dir_parent,
        )

        for succ in successors:
            if succ in closed:
                continue
            g_val = g_scores[succ]
            h_val = octile_distance(succ, goal)
            counter += 1
            heapq.heappush(open_heap, (g_val + h_val, counter, succ[0], succ[1]))

    elapsed_time = time.perf_counter() - start_time
    return [], math.inf, expanded, elapsed_time
