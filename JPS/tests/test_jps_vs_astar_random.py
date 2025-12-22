from __future__ import annotations

from tqdm import tqdm, trange
import math
import numpy as np
import pandas as pd
import random
import unittest
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple

from pathfinding.astar import astar_search
from pathfinding.grid import GridMap
from pathfinding.jps import jump_point_search

from benchmarks.helpers import run_search, save_results


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
    def base_test_random_grids_jps_matches_astar(self, probs, ns, num_samples, num_trials, name) -> None:
        random.seed(0)
        elapsed_times = defaultdict(list)
        expanded_nodes = defaultdict(list)
        for prob in tqdm(probs, desc="Probs", leave=False):
            for n in tqdm(ns, desc="n", leave=False):
                for _ in trange(num_trials, desc="Trials", leave=False):
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
                    for start, goal in tqdm(pairs[:num_samples], desc="Samples", leave=False):
                        path_a, cost_a = run_search(astar_search, "astar", grid, start, goal, elapsed_times, expanded_nodes, prob=prob, n=n)
                        path_j, cost_j = run_search(jump_point_search, "jps", grid, start, goal, elapsed_times, expanded_nodes, prob=prob, n=n)
                        if not path_a:
                            self.assertFalse(path_j, "JPS found a path where A* did not")
                            self.assertTrue(math.isinf(cost_j))
                        else:
                            self.assertTrue(path_j, "JPS failed to find a path that A* found")
                            self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))
        save_results(elapsed_times, expanded_nodes, name)

    def test_random_small(self) -> None:
        self.base_test_random_grids_jps_matches_astar(
            probs=(0.1, 0.25, 0.5, 0.75),
            ns=(8, 12, 20),
            num_trials=10,
            num_samples=10,
            name="jps_vs_astar_random_small",
        )

    def test_random_large(self) -> None:
        self.base_test_random_grids_jps_matches_astar(
            probs=(0.1, 0.25, 0.5, 0.75),
            ns=(256, 512, 1024),
            num_trials=10,
            num_samples=10,
            name="jps_vs_astar_random_large",
        )


if __name__ == "__main__":
    unittest.main()
