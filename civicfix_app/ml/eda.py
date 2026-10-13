"""Build optional NYC 311 chart previews as in-memory data URLs."""
from __future__ import annotations

import base64
from io import BytesIO
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(PROJECT_ROOT, "civicfix_app", "data", "nyc_311_data.csv")


def _image_source(figure):
    output = BytesIO()
    figure.savefig(output, format="png", dpi=160, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


def generate_visualizations():
    """Return two chart previews without writing image files to the project."""
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(f"NYC 311 dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, usecols=["borough", "created_date", "closed_date"], low_memory=False)
    boroughs = df.borough.fillna("Unknown").value_counts().head(8).sort_values()
    created = pd.to_datetime(df.created_date, errors="coerce")
    closed = pd.to_datetime(df.closed_date, errors="coerce")
    hours = (closed - created).dt.total_seconds() / 3600
    hours = hours[(hours > 0) & (hours <= hours.quantile(.95))]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.barh(boroughs.index.str.title(), boroughs.values, color="#6d9b75", height=.62)
    ax.set_title("NYC 311 requests by borough", loc="left", color="#16241e", fontsize=13, weight="bold", pad=14)
    ax.set_xlabel("Service requests", color="#778078")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e9ece8")
    ax.tick_params(axis="y", length=0, colors="#59665d")
    ax.tick_params(axis="x", colors="#89928b")
    ax.grid(axis="x", color="#edf0ec", linewidth=.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    borough_chart = _image_source(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=160)
    ax.hist(hours.dropna(), bins=35, color="#78a193", edgecolor="white", linewidth=.7)
    ax.set_title("Resolution time · closed requests", loc="left", color="#16241e", fontsize=13, weight="bold", pad=14)
    ax.set_xlabel("Hours to resolution", color="#778078")
    ax.set_ylabel("Requests", color="#778078")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#e9ece8")
    ax.spines["left"].set_color("#e9ece8")
    ax.tick_params(colors="#89928b")
    ax.grid(axis="y", color="#edf0ec", linewidth=.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return {"borough_distribution": borough_chart, "resolution_histogram": _image_source(fig)}


if __name__ == "__main__":
    previews = generate_visualizations()
    print(f"Generated {len(previews)} chart previews in memory from nyc_311_data.csv")
