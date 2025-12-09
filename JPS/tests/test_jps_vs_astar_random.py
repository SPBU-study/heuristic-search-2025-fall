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


def get_mean_and_ci95(a: List[float]) -> Tuple[float, float]:
    arr = np.array(a)
    mean = arr.mean()
    stdev = arr.std(ddof=1)
    stderr = stdev / np.sqrt(len(arr))
    ci95 = 1.96 * stderr
    return mean, ci95


class JPSRandomTests(unittest.TestCase):
    def test_random_grids_jps_matches_astar(self) -> None:
        random.seed(0)
        elapsed_times = defaultdict(list)
        expanded_nodes = defaultdict(list)
        num_samples = 30
        for prob in tqdm((0.1, 0.25, 0.5, 0.75), desc="Probs", leave=False):
            for n in tqdm((3, 5, 10, 20, 30), desc="n", leave=False):
                for _ in trange(num_samples, desc="Samples", leave=False):
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
                        path_a, cost_a, expanded_a, elapsed_time_a = astar_search(grid, start, goal)
                        path_j, cost_j, expanded_j, elapsed_time_j = jump_point_search(grid, start, goal)
                        elapsed_times[("astar", prob, n)].append(elapsed_time_a)
                        elapsed_times[("jps", prob, n)].append(elapsed_time_j)
                        expanded_nodes[("astar", prob, n)].append(expanded_a)
                        expanded_nodes[("jps", prob, n)].append(expanded_j)
                        if not path_a:
                            self.assertFalse(path_j, "JPS found a path where A* did not")
                            self.assertTrue(math.isinf(cost_j))
                        else:
                            self.assertTrue(path_j, "JPS failed to find a path that A* found")
                            self.assertTrue(math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6))
        
        res = []
        for (algo, prob, n), times in elapsed_times.items():
            mean_times, ci95_times = get_mean_and_ci95(times)
            mean_expanded, ci95_expanded = get_mean_and_ci95(expanded_nodes[(algo, prob, n)])
            res.append((algo, prob, n, mean_times, ci95_times, mean_expanded, ci95_expanded))
        df = pd.DataFrame(res, columns=["algo", "prob", "n", "mean_times", "ci95_times", "mean_expanded", "ci95_expanded"])
        SAVE_DIR = Path("../artifacts/JPS").resolve()
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SAVE_DIR / "jps_vs_astar_random.csv", index=False)


if __name__ == "__main__":
    unittest.main()
