from typing import List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path


REPO_PATH = Path(__file__).parents[2]


def get_mean_and_ci95(a: List[float]) -> Tuple[float, float]:
    arr = np.array(a)
    mean = arr.mean()
    stdev = arr.std(ddof=1)
    stderr = stdev / np.sqrt(len(arr))
    ci95 = 1.96 * stderr
    return mean, ci95


def run_search(search_func, search_name, grid, start, goal, elapsed_times, expanded_nodes, **kwargs):
    path, cost, expanded, elapsed_time = search_func(grid, start, goal)
    key = " | ".join([search_name] + [f"{x}={y}" for x, y in kwargs.items()])
    elapsed_times[key].append(elapsed_time)
    expanded_nodes[key].append(expanded)
    return path, cost

def save_results(elapsed_times, expanded_nodes, name):
    res = []
    for key, times in elapsed_times.items():
        mean_times, ci95_times = get_mean_and_ci95(times)
        mean_expanded, ci95_expanded = get_mean_and_ci95(expanded_nodes[key])
        search_name = key.split(" | ")[0]
        kvs = "=".join(key.split(" | ")[1:]).split("=")
        ks = kvs[::2]
        vs = kvs[1::2]
        row = dict()
        for k, v in zip(ks, vs):
            row[k] = v
        row.update({
            "search_name": search_name,
            "mean_times": mean_times, 
            "ci95_times": ci95_times, 
            "mean_expanded": mean_expanded, 
            "ci95_expanded": ci95_expanded,
        })
        res.append(row)
    df = pd.DataFrame(res)
    SAVE_DIR = REPO_PATH / "artifacts" / "JPS" / "csvs"
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SAVE_DIR / f"{name}.csv", index=False)