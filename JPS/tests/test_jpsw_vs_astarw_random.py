from __future__ import annotations

import math
import random
import unittest
from typing import List, Tuple

from pathfinding.astarw import astarw_search
from pathfinding.jpsw import jump_point_search_weighted
from pathfinding.weighted_grid import WeightedGridMap


def generate_random_weighted_grid_ascii(
    width: int,
    height: int,
    obstacle_prob: float,
    terrain_symbols: str = "ABCDEF",
) -> WeightedGridMap:
    while True:
        rows: List[str] = []
        for _ in range(height):
            chars_row: List[str] = []
            for _ in range(width):
                if random.random() < obstacle_prob:
                    chars_row.append("#")
                else:
                    chars_row.append(random.choice(terrain_symbols))
            rows.append("".join(chars_row))

        grid = WeightedGridMap.from_ascii(rows)
        free_cells = [
            (x, y)
            for y in range(grid.height)
            for x in range(grid.width)
            if grid.walkable[y][x]
        ]
        if len(free_cells) >= 2:
            return grid


def pick_pairs(grid: WeightedGridMap, count: int) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    free = [(x, y) for y in range(grid.height) for x in range(grid.width) if grid.walkable[y][x]]
    if len(free) < 2:
        return []
    random.shuffle(free)
    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for i in range(0, min(len(free) - 1, 2 * count), 2):
        pairs.append((free[i], free[i + 1]))
    return pairs


class JPSWRandomTests(unittest.TestCase):
    def test_random_grids_jpsw_matches_astarw(self) -> None:
        random.seed(42)
        for size in (8, 12, 20):
            for _ in range(10):
                grid = generate_random_weighted_grid_ascii(size, size, obstacle_prob=0.25)
                pairs = pick_pairs(grid, 5)
                for start, goal in pairs:
                    path_a, cost_a, _ = astarw_search(grid, start, goal)
                    path_j, cost_j, _ = jump_point_search_weighted(grid, start, goal)
                    if not path_a:
                        self.assertFalse(path_j, "JPSW found a path where A*W did not")
                        self.assertTrue(math.isinf(cost_j))
                    else:
                        self.assertTrue(path_j, "JPSW failed to find a path that A*W found")
                        self.assertTrue(
                            math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6),
                            msg=f"Costs differ on size={size}, start={start}, goal={goal}: "
                                f"A*W={cost_a}, JPSW={cost_j}",
                        )


if __name__ == "__main__":
    unittest.main()
