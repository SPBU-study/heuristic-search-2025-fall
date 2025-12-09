from __future__ import annotations

import math
import random
import unittest
from typing import List, Tuple

from pathfinding.astar import astar_search
from pathfinding.grid import GridMap
from pathfinding.jps import jump_point_search


def generate_random_grid(width: int, height: int, block_prob: float) -> Tuple[GridMap, List[str]]:
    while True:
        rows: List[str] = []
        for _ in range(height):
            row_chars = ["." if random.random() > block_prob else "#" for _ in range(width)]
            rows.append("".join(row_chars))
        if sum(row.count(".") for row in rows) >= 2:
            return GridMap.from_ascii(rows), rows


def pick_random_free_cells(grid: GridMap, k: int) -> List[Tuple[int, int]]:
    free = [(x, y) for y in range(grid.height) for x in range(grid.width) if grid.walkable[y][x]]
    if len(free) < k:
        return free
    return random.sample(free, k)


class JPSRandomTests(unittest.TestCase):
    def test_random_grids_jps_matches_astar(self) -> None:
        random.seed(0)
        for prob in (0.1, 0.25, 0.5, 0.75):
            for n in (3, 5, 10, 20, 30):
                for _ in range(100):
                    grid, _ = generate_random_grid(n, n, block_prob=prob)
                    free_cells = [(x, y) for y in range(grid.height) for x in range(grid.width) if grid.walkable[y][x]]
                    if len(free_cells) < 2:
                        continue
                    samples = pick_random_free_cells(grid, min(10, len(free_cells)))
                    if len(samples) < 2:
                        continue
                    pairs = []
                    for i in range(0, len(samples) - 1, 2):
                        pairs.append((samples[i], samples[i + 1]))
                    for start, goal in pairs[:5]:
                        path_a, cost_a, _, _ = astar_search(grid, start, goal)
                        path_j, cost_j, _ = jump_point_search(grid, start, goal)
                        if not path_a:
                            self.assertFalse(path_j, "JPS found a path where A* did not")
                            self.assertTrue(math.isinf(cost_j))
                        else:
                            self.assertTrue(path_j, "JPS failed to find a path that A* found")
                            self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
