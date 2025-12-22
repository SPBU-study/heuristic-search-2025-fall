from __future__ import annotations

import math
import random
import unittest
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict
from pathfinding.astarw import astarw_search
from pathfinding.jpsw import jump_point_search_weighted
from pathfinding.weighted_grid import WeightedGridMap
from tqdm import tqdm, trange
from benchmarks.helpers import run_search, save_results


def _list_weighted_maps(dir_path: Path) -> List[Path]:
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    maps = []
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() == ".map":
            maps.append(p)
    return maps


def _crop_grid(grid: WeightedGridMap, crop_size: int, rng: random.Random) -> Tuple[WeightedGridMap, Tuple[int, int]]:
    cropped, (x0, y0) = grid.random_crop(crop_size, rng=rng, return_offset=True)
    return cropped, (x0, y0)


def _pick_random_pairs(
    grid: WeightedGridMap, k: int, rng: random.Random
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    free = [(x, y) for y in range(grid.height) for x in range(grid.width) if grid.walkable[y][x]]
    if len(free) < 2:
        return []

    pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    for _ in range(k):
        a = rng.choice(free)
        b = rng.choice(free)
        tries = 0
        while b == a and tries < 10:
            b = rng.choice(free)
            tries += 1
        if b != a:
            pairs.append((a, b))
    return pairs


class TestJPSWvsAStarWOnWeightedMaps(unittest.TestCase):
    def test_weighted_maps_crops_jpsw_matches_astarw(self) -> None:
        CROP_SIZE = 128
        N_CROPS_PER_MAP = 3
        N_PAIRS_PER_CROP = 2
        MAX_MAPS = 3
        SEED = 42
        COST_REL_TOL = 1e-6
        COST_ABS_TOL = 1e-6

        elapsed_times = defaultdict(list)
        expanded_nodes = defaultdict(list)

        rng = random.Random(SEED)

        repo_root = Path(__file__).resolve().parents[1]
        maps_dir = repo_root / "maps" / "weighted-map"
        map_paths = _list_weighted_maps(maps_dir)

        if not map_paths:
            self.skipTest(f"No *.map files found in {maps_dir}")

        map_paths = map_paths[:MAX_MAPS]

        for map_path in tqdm(map_paths, desc="Maps", leave=False, total=len(map_paths)):
            grid = WeightedGridMap.from_movingai_map(str(map_path), terrain_weights_path=None)

            if grid.width < CROP_SIZE or grid.height < CROP_SIZE:
                continue

            for crop_idx in trange(N_CROPS_PER_MAP, desc="Crops", leave=False, total=N_CROPS_PER_MAP):
                cropped, (x0, y0) = _crop_grid(grid, CROP_SIZE, rng)
                pairs = _pick_random_pairs(cropped, N_PAIRS_PER_CROP, rng)

                if not pairs:
                    continue

                for (start, goal) in tqdm(pairs, desc="Pairs", leave=False, total=len(pairs)):
                    scen_dir = map_path.parent.stem
                    path_a, cost_a = run_search(astarw_search, "astarw", cropped, start, goal, elapsed_times, expanded_nodes, n=CROP_SIZE, scen_dir=scen_dir)
                    path_j, cost_j = run_search(jump_point_search_weighted, "jpsw", cropped, start, goal, elapsed_times, expanded_nodes, n=CROP_SIZE, scen_dir=scen_dir)

                    ctx = (
                        f"map={map_path.name}, crop={crop_idx}, offset=({x0},{y0}), "
                        f"start={start}, goal={goal}"
                    )

                    if not path_a:
                        self.assertFalse(path_j, "JPSW found path where A*W did not. " + ctx)
                        self.assertTrue(math.isinf(cost_j), "JPSW cost should be inf if no path. " + ctx)
                    else:
                        self.assertTrue(path_j, "JPSW failed to find a path that A*W found. " + ctx)
                        self.assertTrue(
                            math.isclose(cost_a, cost_j, rel_tol=COST_REL_TOL, abs_tol=COST_ABS_TOL),
                            "Costs differ. " + ctx + f", costA={cost_a}, costJ={cost_j}",
                        )
        save_results(elapsed_times, expanded_nodes, "jpsw_vs_astarw_on_movingai")


if __name__ == "__main__":
    unittest.main()
