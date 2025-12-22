from __future__ import annotations

from tqdm import tqdm, trange
import math
import random
import unittest
from collections import defaultdict
from typing import List, Tuple

from pathfinding.astarw import astarw_search
from pathfinding.jpsw import jump_point_search_weighted
from pathfinding.weighted_grid import WeightedGridMap

from benchmarks.helpers import run_search, save_results


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
    def base_test_random_grids_jpsw_matches_astarw(self, probs, ns, num_samples, num_trials, name) -> None:
        random.seed(42)
        elapsed_times = defaultdict(list)
        expanded_nodes = defaultdict(list)
        for prob in tqdm(probs, desc="Probs", leave=False):
            for n in tqdm(ns, desc="n", leave=False):
                for _ in trange(num_trials, desc="Trials", leave=False):
                    grid = generate_random_weighted_grid_ascii(n, n, obstacle_prob=prob)
                    pairs = pick_pairs(grid, num_samples)
                    for start, goal in tqdm(pairs, desc="Samples", leave=False):
                        path_a, cost_a = run_search(astarw_search, "astarw", grid, start, goal, elapsed_times, expanded_nodes, n=n, prob=prob)
                        path_j, cost_j = run_search(jump_point_search_weighted, "jpsw", grid, start, goal, elapsed_times, expanded_nodes, n=n, prob=prob)
                        if not path_a:
                            self.assertFalse(path_j, "JPSW found a path where A*W did not")
                            self.assertTrue(math.isinf(cost_j))
                        else:
                            self.assertTrue(path_j, "JPSW failed to find a path that A*W found")
                            self.assertTrue(
                                math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6),
                                msg=f"Costs differ on size={n}, start={start}, goal={goal}: "
                                    f"A*W={cost_a}, JPSW={cost_j}",
                            )
        save_results(elapsed_times, expanded_nodes, name)

    def test_random_small(self) -> None:
        self.base_test_random_grids_jpsw_matches_astarw(
            probs=(0.1, 0.25, 0.5, 0.75),
            ns=(8, 12, 20),
            num_trials=10,
            num_samples=10,
            name="jpsw_vs_astarw_random_small"
        )

    def test_random_large(self) -> None:
        self.base_test_random_grids_jpsw_matches_astarw(
            probs=(0.1, 0.25, 0.5, 0.75),
            ns=(64, 128, 256),
            num_trials=1,
            num_samples=1,
            name="jpsw_vs_astarw_random_large"
        )


if __name__ == "__main__":
    unittest.main()
