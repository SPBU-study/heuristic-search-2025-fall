from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Tuple

from .astar import astar_search
from .grid import GridMap, ScenarioProblem, load_scenarios
from .jps import jump_point_search, normalize_direction


def _expand_path(path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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


def _render_path(grid: GridMap, path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int]) -> str:
    chars = [row.copy() for row in (grid.chars or [])]
    if not chars:
        chars = [["." if cell else "#" for cell in row] for row in grid.walkable]
    for x, y in path:
        chars[y][x] = "*"
    chars[start[1]][start[0]] = "S"
    chars[goal[1]][goal[0]] = "G"
    return "\n".join("".join(row) for row in chars)


def _run_demo() -> int:
    demo_rows = [
        "..........",
        ".####.....",
        "......#...",
        "...##.....",
        "...##.....",
        "...##.....",
        "...##.....",
        "...##.....",
        "...##.....",
        "..........",
    ]
    grid = GridMap.from_ascii(demo_rows)
    start, goal = (0, 0), (9, 9)
    path, cost, expanded = jump_point_search(grid, start, goal)
    print("Demo grid 10x10")
    print(f"Path cost: {cost:.3f}, expanded nodes: {expanded}")
    if path:
        full_path = _expand_path(path)
        print(_render_path(grid, full_path, start, goal))
        return 0
    print("No path found.")
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run A* or Jump Point Search on MovingAI maps.")
    parser.add_argument("--map", dest="map_path", help="Path to a .map file")
    parser.add_argument("--scenario", dest="scenario_path", help="Path to a .scen file")
    parser.add_argument("--index", type=int, default=0, help="Scenario index (0-based)")
    parser.add_argument("--start-x", type=int, help="Start X coordinate (0-based)")
    parser.add_argument("--start-y", type=int, help="Start Y coordinate (0-based)")
    parser.add_argument("--goal-x", type=int, help="Goal X coordinate (0-based)")
    parser.add_argument("--goal-y", type=int, help="Goal Y coordinate (0-based)")
    parser.add_argument("--algorithm", choices=["jps", "astar"], default="jps", help="Search algorithm")
    parser.add_argument("--show-path", action="store_true", help="Render map with path overlay")
    # parser.add_argument("--max-recursion", type=int, default=10_000, help="Recursion limit for jump search")

    args = parser.parse_args(argv)
    # sys.setrecursionlimit(args.max_recursion)

    if args.map_path is None and args.scenario_path is None:
        return _run_demo()

    start: Tuple[int, int]
    goal: Tuple[int, int]
    grid: GridMap
    optimal_length: Optional[float] = None

    if args.scenario_path:
        problems = load_scenarios(args.scenario_path)
        if not problems:
            print("No scenarios loaded.", file=sys.stderr)
            return 1
        if args.index < 0 or args.index >= len(problems):
            print(f"Scenario index {args.index} out of range (0..{len(problems)-1}).", file=sys.stderr)
            return 1
        prob: ScenarioProblem = problems[args.index]
        grid = GridMap.from_movingai_map(prob.map_path)
        start = (prob.start_x, prob.start_y)
        goal = (prob.goal_x, prob.goal_y)
        optimal_length = prob.optimal_length
    else:
        if args.map_path is None:
            print("Please provide --map or --scenario.", file=sys.stderr)
            return 1
        needed = [args.start_x, args.start_y, args.goal_x, args.goal_y]
        if any(v is None for v in needed):
            print("Start and goal coordinates are required without a scenario.", file=sys.stderr)
            return 1
        grid = GridMap.from_movingai_map(args.map_path)
        start = (int(args.start_x), int(args.start_y))
        goal = (int(args.goal_x), int(args.goal_y))

    if not grid.is_walkable(*start) or not grid.is_walkable(*goal):
        print("Start or goal is blocked.", file=sys.stderr)
        return 1

    if args.algorithm == "jps":
        path, cost, expanded = jump_point_search(grid, start, goal)
        algo_name = "JPS"
    else:
        path, cost, expanded = astar_search(grid, start, goal)
        algo_name = "ASTAR"

    print(f"Algorithm: {algo_name}")
    print(f"Path cost: {cost:.6f}")
    print(f"Expanded nodes: {expanded}")
    if optimal_length is not None:
        diff = abs(cost - optimal_length)
        status = "match" if math.isclose(cost, optimal_length, rel_tol=1e-6, abs_tol=1e-6) else "differs"
        print(f"Scenario optimal: {optimal_length:.6f} ({status}, error {diff:.6f})")

    if args.show_path and path:
        print(_render_path(grid, _expand_path(path), start, goal))
    elif args.show_path:
        print("No path found to display.")

    return 0 if path else 2


if __name__ == "__main__":
    raise SystemExit(main())
