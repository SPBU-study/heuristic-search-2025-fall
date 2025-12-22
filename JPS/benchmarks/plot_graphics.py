import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from typing import List
import click


SAVE_DIR = Path(__file__).parents[2] / "artifacts" / "JPS" / "htmls"


def plot_jps_vs_astar_random(df: pd.DataFrame, metrics: List[str], title: str) -> None:
    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics)
    for i, metric in enumerate(metrics):
        fig_i = px.bar(df, x="label", y=f"mean_{metric}", color="search_name", barmode="group", error_y=f"ci95_{metric}")
        for trace in fig_i.data:
            fig.add_trace(trace, row=i+1, col=1)
    fig.update_layout(title_text=title)
    fig.write_html(SAVE_DIR / f"{title}.html")

@click.command()
@click.option("--input", "-i", type=click.Path(exists=True), required=True)
@click.option("--label", "-l", type=str, required=True, multiple=True)
def main(input, label):
    df = pd.read_csv(input)
    df["label"] = df.apply(lambda row: " | ".join([f"{k}={row[k]}" for k in label]), axis=1)
    title = Path(input).stem
    plot_jps_vs_astar_random(df, ["expanded", "times"], title)


if __name__ == "__main__":
    main()
    