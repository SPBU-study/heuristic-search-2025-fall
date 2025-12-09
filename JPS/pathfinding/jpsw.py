from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Set, Tuple

from .heuristics import DIAGONAL_DISTANCE, weighted_octile_distance
from .jps import DIRECTIONS_8
from .weighted_grid import WeightedGridMap

TIE_EPS = 1e-9


def _lexicographically_better(a: Tuple[float, float], b: Tuple[float, float]) -> bool:
    if a[0] + TIE_EPS < b[0]:
        return True
    if b[0] + TIE_EPS < a[0]:
        return False
    return a[1] + TIE_EPS < b[1]


def _local_patch_nodes(grid: WeightedGridMap, center: Tuple[int, int]) -> Set[Tuple[int, int]]:
    cx, cy = center
    nodes: Set[Tuple[int, int]] = set()
    for y in range(cy - 1, cy + 2):
        for x in range(cx - 1, cx + 2):
            if grid.is_walkable(x, y):
                nodes.add((x, y))
    return nodes


def _local_neighbors(
    grid: WeightedGridMap, node: Tuple[int, int], allowed: Set[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    x, y = node
    res: List[Tuple[int, int]] = []
    for dx, dy in DIRECTIONS_8:
        nx, ny = x + dx, y + dy
        if (nx, ny) not in allowed:
            continue
        if grid.valid_step(x, y, dx, dy):
            res.append((nx, ny))
    return res


def local_dijkstra(
    grid: WeightedGridMap,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    allowed_nodes: Set[Tuple[int, int]],
) -> Tuple[float, float]:
    if start not in allowed_nodes or goal not in allowed_nodes:
        return math.inf, math.inf

    best: Dict[Tuple[int, int], Tuple[float, float]] = {start: (0.0, 0.0)}
    heap: List[Tuple[float, float, int, int]] = [(0.0, 0.0, start[0], start[1])]

    while heap:
        g, last_len, x, y = heapq.heappop(heap)
        recorded = best.get((x, y))
        if recorded is None:
            continue
        if abs(g - recorded[0]) > TIE_EPS or abs(last_len - recorded[1]) > TIE_EPS:
            continue
        if (x, y) == goal:
            return g, last_len

        for nx, ny in _local_neighbors(grid, (x, y), allowed_nodes):
            step_len = DIAGONAL_DISTANCE if nx != x and ny != y else 1.0
            cand = (g + grid.transition_cost(x, y, nx, ny), step_len)
            if _lexicographically_better(cand, best.get((nx, ny), (math.inf, math.inf))):
                best[(nx, ny)] = cand
                heapq.heappush(heap, (cand[0], cand[1], nx, ny))

    return math.inf, math.inf


def _two_step_cost(
    grid: WeightedGridMap, parent: Tuple[int, int], current: Tuple[int, int], neighbor: Tuple[int, int]
) -> Tuple[float, float]:
    dx = neighbor[0] - current[0]
    dy = neighbor[1] - current[1]
    step_len = DIAGONAL_DISTANCE if dx != 0 and dy != 0 else 1.0
    g_px = grid.transition_cost(parent[0], parent[1], current[0], current[1])
    g_xn = grid.transition_cost(current[0], current[1], neighbor[0], neighbor[1])
    return g_px + g_xn, step_len


def prune_neighbors_weighted(
    grid: WeightedGridMap, 
    current: Tuple[int, int], 
    parent: Optional[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    x, y = current
    if parent is None:
        return [(dx, dy) for dx, dy in DIRECTIONS_8 if grid.valid_step(x, y, dx, dy)]

    patch_nodes = _local_patch_nodes(grid, current)
    pruned: List[Tuple[int, int]] = []
    for dx, dy in DIRECTIONS_8:
        if not grid.valid_step(x, y, dx, dy):
            continue
        neighbor = (x + dx, y + dy)
        direct_cost = _two_step_cost(grid, parent, current, neighbor)
        best_cost = local_dijkstra(grid, parent, neighbor, patch_nodes)
        if _lexicographically_better(best_cost, direct_cost):
            continue
        pruned.append((dx, dy))

    return pruned


def _has_multi_terrain_neighbourhood(grid: WeightedGridMap, x: int, y: int) -> bool:
    chars = grid.chars
    seen: set[str] = set()

    for ny in range(y - 1, y + 2):
        for nx in range(x - 1, x + 2):
            if not grid.in_bounds(nx, ny):
                continue
            seen.add(chars[ny][nx])
            if len(seen) > 1:
                return True

    return False


def jump(
    grid: WeightedGridMap, x: int, y: int, dx: int, dy: int, goal: Tuple[int, int]
) -> Optional[Tuple[int, int]]:
    if dx == 0 and dy == 0:
        return None

    cx, cy = x, y
    while True:
        if not grid.valid_step(cx, cy, dx, dy):
            return None

        cx += dx
        cy += dy

        if (cx, cy) == goal:
            return (cx, cy)

        if _has_multi_terrain_neighbourhood(grid, cx, cy):
            return (cx, cy)

        if dx != 0 and dy != 0:
            if jump(grid, cx, cy, dx, 0, goal) is not None:
                return (cx, cy)
            if jump(grid, cx, cy, 0, dy, goal) is not None:
                return (cx, cy)


def _cost_along_ray(
    grid: WeightedGridMap, start: Tuple[int, int], end: Tuple[int, int], dx: int, dy: int
) -> float:
    cx, cy = start
    total = 0.0
    while (cx, cy) != end:
        nx, ny = cx + dx, cy + dy
        total += grid.transition_cost(cx, cy, nx, ny)
        cx, cy = nx, ny
    return total


def identify_successors(
    grid: WeightedGridMap,
    current: Tuple[int, int],
    prune_parent: Optional[Tuple[int, int]],
    goal: Tuple[int, int],
    g_scores: Dict[Tuple[int, int], float],
    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
    dir_parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    successors: List[Tuple[int, int]] = []
    x, y = current
    directions = prune_neighbors_weighted(grid, current, prune_parent)

    for dx, dy in directions:
        jp = jump(grid, x, y, dx, dy, goal)
        if jp is None:
            continue

        jx, jy = jp
        move_cost = _cost_along_ray(grid, (x, y), (jx, jy), dx, dy)
        tentative_g = g_scores[current] + move_cost

        if tentative_g + 1e-9 < g_scores.get((jx, jy), math.inf):
            g_scores[(jx, jy)] = tentative_g
            parent_map[(jx, jy)] = current
            dir_parent[(jx, jy)] = (jx - dx, jy - dy)
            successors.append((jx, jy))

    return successors


def _reconstruct_path(
    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]], goal: Tuple[int, int]
) -> List[Tuple[int, int]]:
    path: List[Tuple[int, int]] = []
    node: Optional[Tuple[int, int]] = goal
    while node is not None:
        path.append(node)
        node = parent_map.get(node)
    path.reverse()
    return path


def jump_point_search_weighted(
    grid: WeightedGridMap, start: Tuple[int, int], goal: Tuple[int, int]
) -> Tuple[List[Tuple[int, int]], float, int]:
    open_heap: List[Tuple[float, int, int, int]] = []
    g_scores: Dict[Tuple[int, int], float] = {start: 0.0}
    parent_map: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    dir_parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
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
            return _reconstruct_path(parent_map, goal), g_current, expanded

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
            h_val = weighted_octile_distance(succ, goal, grid)
            counter += 1
            heapq.heappush(open_heap, (g_val + h_val, counter, succ[0], succ[1]))

    return [], math.inf, expanded
