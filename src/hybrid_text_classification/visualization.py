"""Visualization utilities for ArtSense ML datasets and experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .constants import DEFAULT_DATA_PATH, NUMERIC_COLUMNS
from .preprocessing import clean_data

NUMERIC_FEATURE_SHORT_TITLES: list[str] = [
    "Emotion Intensity",
    "Feel Sombre",
    "Feel Content",
    "Feel Calm",
    "Feel Uneasy",
    "Num Colours",
    "Num Objects",
    "Willingness to Pay",
]


def plot_numeric_feature_distributions(
    data_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = "reports/figures/data_distribution.png",
    bins: int = 10,
) -> None:
    """Save histogram grid showing the distribution of numeric input features.

    Args:
        data_path: Path to the raw project CSV.
        output_path: File path where the PNG visualization should be saved.
        bins: Number of histogram bins for each numeric feature.
    """
    df: pd.DataFrame = clean_data(data_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for i, ax in enumerate(axes.flatten()):
        ax.hist(df[NUMERIC_COLUMNS[i]], bins=bins)
        ax.set_title(NUMERIC_FEATURE_SHORT_TITLES[i], fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

    plt.suptitle("Distribution of Numeric Features", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create ArtSense ML data visualizations.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to project CSV.")
    parser.add_argument(
        "--output",
        default="reports/figures/data_distribution.png",
        help="Where to save the numeric feature distribution plot.",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of histogram bins.")
    args = parser.parse_args()

    plot_numeric_feature_distributions(
        data_path=args.data,
        output_path=args.output,
        bins=args.bins,
    )
    print(f"Saved numeric feature distribution plot to {args.output}")


if __name__ == "__main__":
    main()
