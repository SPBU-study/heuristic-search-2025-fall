from __future__ import annotations

import math
import unittest
from typing import Dict, List, Tuple

from pathfinding.astarw import astarw_search
from pathfinding.jpsw import jump_point_search_weighted
from pathfinding.weighted_grid import WeightedGridMap


def weighted_grid(rows: List[str], mapping: Dict[str, float]) -> WeightedGridMap:
    return WeightedGridMap.from_ascii(rows, weight_mapping=mapping)


class JPSWSmallTests(unittest.TestCase):
    def assert_costs_match(self, grid: WeightedGridMap, start: Tuple[int, int], goal: Tuple[int, int]) -> None:
        path_a, cost_a, _ = astarw_search(grid, start, goal)
        path_j, cost_j, _ = jump_point_search_weighted(grid, start, goal)
        if not path_a:
            self.assertFalse(path_j, "JPSW found a path where A*W did not")
            self.assertTrue(math.isinf(cost_j))
            return
        self.assertTrue(path_j, "JPSW failed to find a path A*W found")
        self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))

    def test_uniform_weights(self) -> None:
        rows = ["....."] * 5
        mapping = {".": 1.0}
        grid = weighted_grid(rows, mapping)
        start, goal = (0, 0), (4, 4)
        path_a, cost_a, _ = astarw_search(grid, start, goal)
        path_j, cost_j, _ = jump_point_search_weighted(grid, start, goal)
        self.assertTrue(path_a and path_j)
        self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))

    def test_heavy_center_prefers_detour(self) -> None:
        rows = [
            "...",
            ".H.",
            "...",
        ]
        mapping = {".": 1.0, "H": 10.0}
        grid = weighted_grid(rows, mapping)
        start, goal = (0, 1), (2, 1)
        path_a, cost_a, _ = astarw_search(grid, start, goal)
        self.assertTrue(path_a)
        self.assertLess(cost_a, 6.0)
        self.assert_costs_match(grid, start, goal)

    def test_start_equals_goal(self) -> None:
        rows = [
            "...",
            "...",
            "...",
        ]
        mapping = {".": 1.0}
        grid = weighted_grid(rows, mapping)
        start = goal = (1, 1)
        path_a, cost_a, _ = astarw_search(grid, start, goal)
        path_j, cost_j, _ = jump_point_search_weighted(grid, start, goal)
        self.assertEqual(path_a, [(1, 1)])
        self.assertEqual(path_j, [(1, 1)])
        self.assertTrue(math.isclose(cost_a, 0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(cost_j, 0.0, abs_tol=1e-9))

    def test_no_path_blocked_regions(self) -> None:
        rows = [
            "##.##",
            "#####",
            "##.##",
        ]
        mapping = {".": 1.0}
        grid = weighted_grid(rows, mapping)
        start, goal = (2, 0), (2, 2)
        self.assert_costs_match(grid, start, goal)


if __name__ == "__main__":
    unittest.main()
