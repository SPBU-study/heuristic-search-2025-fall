import pandas as pd
import plotly.express as px
from pathlib import Path


READ_DIR = Path("../artifacts/JPS").resolve()

def plot_jps_vs_astar_random_expanded(df: pd.DataFrame) -> None:
    fig = px.bar(df, x="label", y="mean_expanded", color="algo", barmode="group", error_y="ci95_expanded")
    fig.write_html(READ_DIR / "jps_vs_astar_random_expanded.html")

def plot_jps_vs_astar_random_time(df: pd.DataFrame) -> None:
    fig = px.bar(df, x="label", y="mean_times", color="algo", barmode="group", error_y="ci95_times")
    fig.write_html(READ_DIR / "jps_vs_astar_random_time.html")

def plot_jps_vs_astar_movingai_scen_expanded(df: pd.DataFrame) -> None:
    fig = px.bar(df, x="label", y="mean_expanded", color="algo", barmode="group", error_y="ci95_expanded")
    fig.write_html(READ_DIR / "jps_vs_astar_movingai_scen__largest_expanded.html")

def plot_jps_vs_astar_movingai_scen_time(df: pd.DataFrame) -> None:
    fig = px.bar(df, x="label", y="mean_times", color="algo", barmode="group", error_y="ci95_times")
    fig.write_html(READ_DIR / "jps_vs_astar_movingai_scen__largest_time.html")

if __name__ == "__main__":
    # df = pd.read_csv(READ_DIR / "jps_vs_astar_random.csv")
    # df["label"] = df.apply(lambda row: f"{row['prob']} {row['n']}", axis=1)
    # plot_jps_vs_astar_random_expanded(df)
    # plot_jps_vs_astar_random_time(df)
    df = pd.read_csv(READ_DIR / "jps_vs_astar_movingai_scen__largest.csv")
    df["label"] = df.apply(lambda row: f"{row['scen_name']} {row['n']}", axis=1)
    plot_jps_vs_astar_movingai_scen_expanded(df)
    plot_jps_vs_astar_movingai_scen_time(df)
