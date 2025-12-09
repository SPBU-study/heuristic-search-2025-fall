from __future__ import annotations

import math
import unittest
from typing import List, Tuple

from pathfinding.astar import astar_search
from pathfinding.grid import GridMap
from pathfinding.jps import jump_point_search


def grid_from_ascii(rows: List[str]) -> GridMap:
    return GridMap.from_ascii(rows)


class JPSSmallTests(unittest.TestCase):
    def assert_costs_match(self, grid: GridMap, start: Tuple[int, int], goal: Tuple[int, int]) -> None:
        path_a, cost_a, _, _ = astar_search(grid, start, goal)
        path_j, cost_j, _ = jump_point_search(grid, start, goal)
        if not path_a:
            self.assertFalse(path_j, "JPS found a path where A* did not")
            self.assertTrue(math.isinf(cost_j))
            return
        self.assertTrue(path_j, "JPS failed to find a path A* found")
        self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))

    def test_simple_open_grid_diagonal(self) -> None:
        grid = grid_from_ascii(["....."] * 5)
        start, goal = (0, 0), (4, 4)
        path_a, cost_a, _, _ = astar_search(grid, start, goal)
        path_j, cost_j, _ = jump_point_search(grid, start, goal)
        self.assertTrue(path_a and path_j)
        self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))

    def test_corridor_with_central_block_regression(self) -> None:
        rows = [
            "...",
            ".#.",
            "...",
        ]
        grid = grid_from_ascii(rows)
        start, goal = (0, 1), (2, 1)
        path_a, cost_a, _, _ = astar_search(grid, start, goal)
        path_j, cost_j, _ = jump_point_search(grid, start, goal)
        self.assertTrue(path_a)
        self.assertTrue(path_j, "Regression: JPS previously failed here")
        self.assertTrue(math.isclose(cost_a, 4.0, abs_tol=1e-6))
        self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))

    def test_no_path_blocked_goal(self) -> None:
        rows = [
            "#####",
            "#.#.#",
            "#.#.#",
            "#.#.#",
            "#####",
        ]
        grid = grid_from_ascii(rows)
        start, goal = (1, 1), (3, 3)
        path_a, cost_a, _, _ = astar_search(grid, start, goal)
        path_j, cost_j, _ = jump_point_search(grid, start, goal)
        self.assertFalse(path_a)
        self.assertFalse(path_j)
        self.assertTrue(math.isinf(cost_a))
        self.assertTrue(math.isinf(cost_j))

    def test_turning_corridor(self) -> None:
        rows = [
            ".....",
            "###..",
            ".....",
        ]
        grid = grid_from_ascii(rows)
        start, goal = (0, 0), (0, 2)
        self.assert_costs_match(grid, start, goal)


    def test_dead_end_horizontal_corridor_turn_up(self) -> None:
        rows = [
            "###.#",
            "....#",
            "#####",
        ]
        grid = grid_from_ascii(rows)
        start = (0, 1)
        goal = (3, 0)
        self.assert_costs_match(grid, start, goal)

    def test_dead_end_horizontal_corridor_turn_down(self) -> None:
        rows = [
            "#####",
            "....#",
            "###.#",
        ]
        grid = grid_from_ascii(rows)
        start = (0, 1)
        goal = (3, 2)
        self.assert_costs_match(grid, start, goal)

    def test_dead_end_vertical_corridor_turn_right(self) -> None:
        rows = [
            "###",
            ".#.",
            ".#.",
            ".#.",
            ".##",
            ".#.",
            "###",
        ]

        grid = grid_from_ascii(rows)
        start = (0, 1)
        goal = (1, 4)
        self.assert_costs_match(grid, start, goal)

    def test_cross_shaped_trap_around_central_block(self) -> None:
        rows = [
            ".....",
            ".###.",
            ".#.#.",
            ".###.",
            ".....",
        ]
        grid = grid_from_ascii(rows)
        start = (0, 2)
        goal = (4, 2)
        self.assert_costs_match(grid, start, goal)

    def test_zigzag_corridor_with_side_pockets(self) -> None:
        rows = [
            "########",
            "#......#",
            "#.####.#",
            "#.#..#.#",
            "#.#.##.#",
            "#.#....#",
            "########",
        ]
        grid = grid_from_ascii(rows)
        start = (1, 1)
        goal = (4, 3)
        self.assert_costs_match(grid, start, goal)

    def test_mini(self) -> None:
        rows = [
            "#..",
            "...",
            "...",
        ]
        grid = grid_from_ascii(rows)
        start = (1, 0)
        goal = (0, 2)
        self.assert_costs_match(grid, start, goal)

if __name__ == "__main__":
    unittest.main()
