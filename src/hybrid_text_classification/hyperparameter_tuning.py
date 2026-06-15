"""Hyperparameter tuning experiments for ArtSense ML."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .constants import DEFAULT_DATA_PATH
from .features import build_training_features
from .model import LogisticRegression, accuracy
from .preprocessing import clean_data, split_data


def prepare_data(data_path: str | Path = DEFAULT_DATA_PATH, min_freq: int = 5) -> dict[str, object]:
    """Prepare train/validation features for hyperparameter experiments."""
    df = clean_data(data_path)
    train_df, valid_df, test_df = split_data(df)
    return build_training_features(train_df, valid_df, test_df, minimum_frequency=min_freq)


def _save_line_plot(
    x_values: Iterable[float | int],
    scores: list[float],
    xlabel: str,
    title: str,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 4))
    plt.plot(list(x_values), scores, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Validation Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def tune_learning_rates(
    data_path: str | Path = DEFAULT_DATA_PATH,
    learning_rates: list[float] | None = None,
    output_path: str | Path = "reports/figures/lr_tuning.png",
) -> pd.DataFrame:
    """Evaluate validation accuracy across learning rates."""
    learning_rates = learning_rates or [0.01, 0.05, 0.1]
    features = prepare_data(data_path=data_path, min_freq=5)
    scores: list[float] = []

    for lr in learning_rates:
        model = LogisticRegression(lr=lr, n_iters=3000, random_state=0)
        model.fit(features["X_train"], features["t_train"])
        score = accuracy(model.predict_class(features["X_valid"]), features["t_valid"])
        scores.append(score)
        print(f"lr={lr}: valid acc={score:.4f}")

    _save_line_plot(
        learning_rates,
        scores,
        xlabel="Learning Rate",
        title="Validation Accuracy vs Learning Rate",
        output_path=output_path,
    )
    return pd.DataFrame({"learning_rate": learning_rates, "validation_accuracy": scores})


def tune_iterations(
    data_path: str | Path = DEFAULT_DATA_PATH,
    iteration_counts: list[int] | None = None,
    output_path: str | Path = "reports/figures/iters_tuning.png",
) -> pd.DataFrame:
    """Evaluate validation accuracy across gradient descent iteration counts."""
    iteration_counts = iteration_counts or [1000, 2000, 3000]
    features = prepare_data(data_path=data_path, min_freq=5)
    scores: list[float] = []

    for n_iters in iteration_counts:
        model = LogisticRegression(lr=0.05, n_iters=n_iters, random_state=0)
        model.fit(features["X_train"], features["t_train"])
        score = accuracy(model.predict_class(features["X_valid"]), features["t_valid"])
        scores.append(score)
        print(f"n_iters={n_iters}: valid acc={score:.4f}")

    _save_line_plot(
        iteration_counts,
        scores,
        xlabel="Number of Iterations",
        title="Validation Accuracy vs Number of Iterations",
        output_path=output_path,
    )
    return pd.DataFrame({"n_iters": iteration_counts, "validation_accuracy": scores})


def tune_min_frequency(
    data_path: str | Path = DEFAULT_DATA_PATH,
    min_frequencies: list[int] | None = None,
    output_path: str | Path = "reports/figures/minfreq_tuning.png",
) -> pd.DataFrame:
    """Evaluate validation accuracy across vocabulary minimum-frequency cutoffs."""
    min_frequencies = min_frequencies or [3, 5, 10]
    scores: list[float] = []

    for min_freq in min_frequencies:
        features = prepare_data(data_path=data_path, min_freq=min_freq)
        model = LogisticRegression(lr=0.05, n_iters=3000, random_state=0)
        model.fit(features["X_train"], features["t_train"])
        score = accuracy(model.predict_class(features["X_valid"]), features["t_valid"])
        scores.append(score)
        print(f"min_freq={min_freq}: valid acc={score:.4f}")

    _save_line_plot(
        min_frequencies,
        scores,
        xlabel="Minimum Word Frequency",
        title="Validation Accuracy vs Minimum Word Frequency",
        output_path=output_path,
    )
    return pd.DataFrame({"min_freq": min_frequencies, "validation_accuracy": scores})


def tune_l2_lambda(
    data_path: str | Path = DEFAULT_DATA_PATH,
    lambdas: list[float] | None = None,
    output_path: str | Path = "reports/figures/lambda_tuning.png",
) -> pd.DataFrame:
    """Evaluate validation accuracy across L2 regularization strengths."""
    lambdas = lambdas or [0, 0.01, 0.05, 0.1, 0.2, 0.5]
    features = prepare_data(data_path=data_path, min_freq=5)
    scores: list[float] = []

    for l2_lambda in lambdas:
        model = LogisticRegression(lr=0.05, n_iters=3000, l2_lambda=l2_lambda, random_state=0)
        model.fit(features["X_train"], features["t_train"])
        score = accuracy(model.predict_class(features["X_valid"]), features["t_valid"])
        scores.append(score)
        print(f"lambda={l2_lambda}: valid acc={score:.4f}")

    _save_line_plot(
        lambdas,
        scores,
        xlabel="Lambda (Regularization Strength)",
        title="Validation Accuracy vs Lambda",
        output_path=output_path,
    )
    return pd.DataFrame({"l2_lambda": lambdas, "validation_accuracy": scores})


def run_all_tuning(data_path: str | Path = DEFAULT_DATA_PATH, output_dir: str | Path = "reports/figures") -> dict[str, pd.DataFrame]:
    """Run all hyperparameter experiments and save their plots."""
    output_dir = Path(output_dir)
    return {
        "learning_rates": tune_learning_rates(data_path, output_path=output_dir / "lr_tuning.png"),
        "iterations": tune_iterations(data_path, output_path=output_dir / "iters_tuning.png"),
        "min_frequency": tune_min_frequency(data_path, output_path=output_dir / "minfreq_tuning.png"),
        "l2_lambda": tune_l2_lambda(data_path, output_path=output_dir / "lambda_tuning.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArtSense ML hyperparameter tuning experiments.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to project CSV.")
    parser.add_argument("--output-dir", default="reports/figures", help="Directory to save tuning plots.")
    parser.add_argument(
        "--experiment",
        choices=["all", "lr", "iters", "minfreq", "lambda"],
        default="all",
        help="Which tuning experiment to run.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.experiment == "all":
        run_all_tuning(args.data, output_dir)
    elif args.experiment == "lr":
        tune_learning_rates(args.data, output_path=output_dir / "lr_tuning.png")
    elif args.experiment == "iters":
        tune_iterations(args.data, output_path=output_dir / "iters_tuning.png")
    elif args.experiment == "minfreq":
        tune_min_frequency(args.data, output_path=output_dir / "minfreq_tuning.png")
    elif args.experiment == "lambda":
        tune_l2_lambda(args.data, output_path=output_dir / "lambda_tuning.png")


if __name__ == "__main__":
    main()
