"""Model comparison experiments for ArtSense ML."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

from .features import build_training_features
from .model import LogisticRegression, accuracy
from .preprocessing import clean_data, split_data


def run_logistic_regression(features: dict[str, object], seed: int) -> float:
    model = LogisticRegression(lr=0.05, n_iters=3000, random_state=seed)
    model.fit(features["X_train"], features["t_train"])
    return accuracy(model.predict_class(features["X_valid"]), features["t_valid"])


def run_random_forest(features: dict[str, object], seed: int) -> float:
    X_train = np.hstack([features["X_train_bow_raw"], features["X_train_num_raw"]])
    X_valid = np.hstack([features["X_valid_bow_raw"], features["X_valid_num_raw"]])
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train, features["t_train"])
    return accuracy(model.predict(X_valid), features["t_valid"])


def run_decision_tree(features: dict[str, object], seed: int) -> float:
    model = DecisionTreeClassifier(random_state=seed, max_depth=5)
    model.fit(features["X_train_num_raw"], features["t_train"])
    return accuracy(model.predict(features["X_valid_num_raw"]), features["t_valid"])


def run_rf_plus_nb(features: dict[str, object], seed: int) -> float:
    nb = MultinomialNB()
    nb.fit(features["X_train_bow_raw"], features["t_train"])

    train_nb_probs = nb.predict_proba(features["X_train_bow_raw"])
    valid_nb_probs = nb.predict_proba(features["X_valid_bow_raw"])

    X_train_rf = np.hstack([features["X_train_num_raw"], train_nb_probs])
    X_valid_rf = np.hstack([features["X_valid_num_raw"], valid_nb_probs])

    rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    rf.fit(X_train_rf, features["t_train"])
    return accuracy(rf.predict(X_valid_rf), features["t_valid"])


def compare_models(data_path: str | Path, n_runs: int = 10, output_path: str | Path = "model_comparison.png") -> pd.DataFrame:
    """Run repeated validation comparisons and save a bar chart."""
    df = clean_data(data_path)
    results = []

    for seed in range(n_runs):
        train_df, valid_df, test_df = split_data(df, random_state=seed)
        features = build_training_features(train_df, valid_df, test_df, minimum_frequency=5)

        results.append({
            "seed": seed,
            "Logistic Regression": run_logistic_regression(features, seed),
            "Random Forest": run_random_forest(features, seed),
            "RF + NB": run_rf_plus_nb(features, seed),
            "Decision Tree": run_decision_tree(features, seed),
        })

    results_df = pd.DataFrame(results)
    model_names = ["Logistic Regression", "Random Forest", "RF + NB", "Decision Tree"]
    means = [results_df[name].mean() for name in model_names]
    stds = [results_df[name].std() for name in model_names]

    plt.figure(figsize=(9, 5))
    plt.bar(model_names, means, yerr=stds, capsize=5)
    plt.ylabel("Validation Accuracy")
    plt.title(f"Model Comparison Over {n_runs} Runs")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)

    return results_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ArtSense ML model variants.")
    parser.add_argument("--data", required=True, help="Path to cleaned/raw training CSV.")
    parser.add_argument("--runs", type=int, default=10, help="Number of random seeds to evaluate.")
    parser.add_argument("--output", default="model_comparison.png", help="Output chart path.")
    args = parser.parse_args()

    results = compare_models(args.data, n_runs=args.runs, output_path=args.output)
    print(results)
    print("\nMean validation accuracy:")
    print(results.drop(columns=["seed"]).agg(["mean", "std"]).T)


if __name__ == "__main__":
    main()
