from __future__ import annotations

import math
import os
import unittest
from typing import Iterable, List

from pathfinding.astar import astar_search
from pathfinding.grid import GridMap, load_scenarios
from pathfinding.jps import jump_point_search

MAPS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps")

SCEN_DIRS: List[str] = [
    "maze-scen",
    "random-scen",
    "room-scen",
]

MAX_PROBLEMS_PER_SCEN = 2


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
        for scen_path in _iter_scen_files():
            problems = load_scenarios(scen_path)
            if not problems:
                continue

            if MAX_PROBLEMS_PER_SCEN is not None:
                problems = problems[:MAX_PROBLEMS_PER_SCEN]

            for idx, prob in enumerate(problems):
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

                    path_a, cost_a, _, _ = astar_search(grid, start, goal)
                    path_j, cost_j, _, _ = jump_point_search(grid, start, goal)

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


if __name__ == "__main__":
    unittest.main()
