"""Training script for ArtSense ML."""

from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import save_artifacts
from .constants import DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR
from .features import build_training_features
from .model import LogisticRegression, accuracy, cross_entropy_loss
from .preprocessing import clean_data, split_data


def train(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    lr: float = 0.05,
    n_iters: int = 3000,
    l2_lambda: float = 0.5,
    min_freq: int = 5,
    random_state: int = 42,
) -> dict[str, float]:
    """Train the model and save learned artifacts."""
    df = clean_data(data_path)
    train_df, valid_df, test_df = split_data(df, random_state=1)
    features = build_training_features(train_df, valid_df, test_df, minimum_frequency=min_freq)

    model = LogisticRegression(
        lr=lr,
        n_iters=n_iters,
        l2_lambda=l2_lambda,
        random_state=random_state,
    )
    model.fit(features["X_train"], features["t_train"])

    save_artifacts(
        model=model,
        vocab=features["vocab"],
        bow_stats=features["bow_stats"],
        num_stats=features["num_stats"],
        model_dir=model_dir,
    )

    metrics = {}
    for split_name in ["train", "valid", "test"]:
        X = features[f"X_{split_name}"]
        t = features[f"t_{split_name}"]
        probs = model.predict_proba(X)
        preds = model.predict_class(X)
        metrics[f"{split_name}_loss"] = cross_entropy_loss(probs, t)
        metrics[f"{split_name}_accuracy"] = accuracy(preds, t)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ArtSense ML.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, help="Path to training CSV.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Directory for saved artifacts.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--n-iters", type=int, default=3000, help="Maximum gradient descent iterations.")
    parser.add_argument("--l2", type=float, default=0.5, help="L2 regularization strength.")
    parser.add_argument("--min-freq", type=int, default=5, help="Minimum word frequency for vocabulary.")
    args = parser.parse_args()

    metrics = train(
        data_path=args.data,
        model_dir=args.model_dir,
        lr=args.lr,
        n_iters=args.n_iters,
        l2_lambda=args.l2,
        min_freq=args.min_freq,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
