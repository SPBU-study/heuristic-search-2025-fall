import math
import os
import unittest
from typing import Iterable, List, Tuple
from collections import defaultdict
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm, trange   
from typing import Union, Literal
import random

from pathfinding.astarw import astarw_search
from pathfinding.grid import GridMap, load_scenarios
from pathfinding.jpsw import jump_point_search_weighted
from pathfinding.weighted_grid import WeightedGridMap

from benchmarks.helpers import run_search, save_results


MAPS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps")

SCEN_DIRS: List[str] = [
    "weighted-scen",
]
TERRAIN_WEIGHTS_PATH: str = os.path.join(MAPS_ROOT, "terrain_weights.json")

MAX_PROBLEMS_PER_SCEN = 1


def _iter_scen_files() -> Iterable[str]:
    for scen_dir in SCEN_DIRS:
        full_dir = os.path.join(MAPS_ROOT, scen_dir)
        if not os.path.isdir(full_dir):
            continue
        for name in os.listdir(full_dir):
            if name.lower().endswith(".scen"):
                yield os.path.join(full_dir, name)


class MovingAIScenBenchmarkTests(unittest.TestCase):
    def base_test_all_scenarios_against_astar(self, sampling: str = Union[Literal["largest"], Literal["smallest"], Literal["random"]]) -> None:
        total_checked = 0
        elapsed_times = defaultdict(list)
        expanded_nodes = defaultdict(list)
        files = list(_iter_scen_files())
        sampled_files = random.sample(files, 1)
        for scen_path in tqdm(sampled_files, desc="Scenarios", leave=False, total=len(sampled_files)):
            scen_dir = Path(scen_path).parent.stem.split("-")[0]
            problems = load_scenarios(scen_path)
            if not problems:
                continue

            if MAX_PROBLEMS_PER_SCEN is not None:
                if sampling == "largest":
                    problems = problems[-MAX_PROBLEMS_PER_SCEN:]
                elif sampling == "smallest":
                    problems = problems[:MAX_PROBLEMS_PER_SCEN]
                elif sampling == "random":
                    problems = random.sample(problems, MAX_PROBLEMS_PER_SCEN)

            for idx, prob in tqdm(enumerate(problems), desc="Problems", leave=False, total=len(problems)):
                with self.subTest(scenario=scen_path, index=idx):
                    grid = WeightedGridMap.from_movingai_map(prob.map_path, terrain_weights_path=TERRAIN_WEIGHTS_PATH)
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

                    n = grid.height
                    path_a, cost_a = run_search(astarw_search, "astarw", grid, start, goal, elapsed_times, expanded_nodes, n=n, scen_dir=scen_dir)
                    path_j, cost_j = run_search(jump_point_search_weighted, "jpsw", grid, start, goal, elapsed_times, expanded_nodes, n=n, scen_dir=scen_dir)
                    if not path_a:
                        self.assertFalse(
                            path_j,
                            f"JPSW нашёл путь, а A*W нет ({scen_path}, index={idx})",
                        )
                        self.assertTrue(
                            math.isinf(cost_j),
                            "JPSW должен вернуть бесконечную стоимость, если пути нет",
                        )
                    else:
                        self.assertTrue(
                            path_j,
                            f"JPSW не нашёл путь, который A*W нашёл ({scen_path}, index={idx})",
                        )
                        self.assertTrue(
                            math.isclose(cost_a, cost_j, rel_tol=1e-6, abs_tol=1e-6),
                            f"JPSW дал другую стоимость: A*W={cost_a}, JPSW={cost_j} "
                            f"({scen_path}, index={idx})",
                        )

                    total_checked += 1

        self.assertGreater(total_checked, 0, "Не удалось найти ни одного сценария для теста")
        save_results(elapsed_times, expanded_nodes, f"jpsw_vs_astarw_movingai_scen__{sampling}")

    def test_largest(self) -> None:
        self.base_test_all_scenarios_against_astar(sampling="largest")
    
    def test_smallest(self) -> None:
        self.base_test_all_scenarios_against_astar(sampling="smallest")
    
    def test_random(self) -> None:
        self.base_test_all_scenarios_against_astar(sampling="random")

if __name__ == "__main__":
    unittest.main()
