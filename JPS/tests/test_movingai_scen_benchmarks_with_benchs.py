from __future__ import annotations

import math
import os
import unittest
from typing import Iterable, List, Tuple
from collections import defaultdict
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange

from pathfinding.astar import astar_search
from pathfinding.grid import GridMap, load_scenarios
from pathfinding.jps import jump_point_search

MAPS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps")

SCEN_DIRS: List[str] = [
    "maze-scen",
    # "random-scen",
    # "room-scen",
]

MAX_PROBLEMS_PER_SCEN = 2

def get_mean_and_ci95(a: List[float]) -> Tuple[float, float]:
    arr = np.array(a)
    mean = arr.mean()
    stdev = arr.std(ddof=1)
    stderr = stdev / np.sqrt(len(arr))
    ci95 = 1.96 * stderr
    return mean, ci95

def _iter_scen_files() -> Iterable[str]:
    for scen_dir in SCEN_DIRS:
        full_dir = os.path.join(MAPS_ROOT, scen_dir)
        if not os.path.isdir(full_dir):
            continue
        for name in os.listdir(full_dir):
            if name.lower().endswith(".scen"):
                yield os.path.join(full_dir, name)


class MovingAIScenBenchmarkTests(unittest.TestCase):
    def test_all_scenarios_against_astar(self) -> None:
        total_checked = 0
        elapsed_times = defaultdict(list)
        expanded_nodes = defaultdict(list)
        for scen_path in tqdm(_iter_scen_files(), desc="Scenarios", leave=False, total=60):
            problems = load_scenarios(scen_path)
            if not problems:
                continue

            if MAX_PROBLEMS_PER_SCEN is not None:
                # problems = problems[-MAX_PROBLEMS_PER_SCEN:]
                problems = problems[:MAX_PROBLEMS_PER_SCEN]

            for idx, prob in tqdm(enumerate(problems), desc="Problems", leave=False, total=len(problems)):
                with self.subTest(scenario=scen_path, index=idx):
                    grid = GridMap.from_movingai_map(prob.map_path)
                    self.assertEqual(
                        grid.width,
                        prob.map_width,
                        msg=f"Width mismatch for {scen_path} #{idx}",
                    )
                    self.assertEqual(
                        grid.height,
                        prob.map_height,
                        msg=f"Height mismatch for {scen_path} #{idx}",
                    )

                    start = (prob.start_x, prob.start_y)
                    goal = (prob.goal_x, prob.goal_y)

                    path_a, cost_a, expanded_a, elapsed_time_a = astar_search(grid, start, goal)
                    path_j, cost_j, expanded_j, elapsed_time_j = jump_point_search(grid, start, goal)
                    elapsed_times[("astar", grid.height, Path(scen_path).stem)].append(elapsed_time_a)
                    elapsed_times[("jps", grid.height, Path(scen_path).stem)].append(elapsed_time_j)
                    expanded_nodes[("astar", grid.height, Path(scen_path).stem)].append(expanded_a)
                    expanded_nodes[("jps", grid.height, Path(scen_path).stem)].append(expanded_j)

                    if not path_a:
                        self.assertFalse(
                            path_j,
                            f"JPS нашёл путь, а A* нет ({scen_path}, index={idx})",
                        )
                        self.assertTrue(
                            math.isinf(cost_j),
                            "JPS должен вернуть бесконечную стоимость, если пути нет",
                        )
                    else:
                        self.assertTrue(
                            path_j,
                            f"JPS не нашёл путь, который A* нашёл ({scen_path}, index={idx})",
                        )
                        self.assertTrue(
                            math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6),
                            f"JPS дал другую стоимость: A*={cost_a}, JPS={cost_j} "
                            f"({scen_path}, index={idx})",
                        )

                    total_checked += 1

        self.assertGreater(total_checked, 0, "Не удалось найти ни одного сценария для теста")
        res = []
        for (algo, height, scen_name), times in elapsed_times.items():
            mean_times, ci95_times = get_mean_and_ci95(times)
            mean_expanded, ci95_expanded = get_mean_and_ci95(expanded_nodes[(algo, height, scen_name)])
            res.append((algo, height, scen_name, mean_times, ci95_times, mean_expanded, ci95_expanded))
        df = pd.DataFrame(res, columns=["algo", "n", "scen_name", "mean_times", "ci95_times", "mean_expanded", "ci95_expanded"])
        SAVE_DIR = Path("../artifacts/JPS").resolve()
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(SAVE_DIR / "jps_vs_astar_movingai_scen__largest.csv", index=False)

if __name__ == "__main__":
    unittest.main()
