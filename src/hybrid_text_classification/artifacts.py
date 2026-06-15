"""Helpers for saving/loading model artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import NormalizationStats
from .model import LogisticRegression


def load_artifacts(model_dir: str | Path) -> dict[str, object]:
    """Load saved model weights, vocabulary, and normalization statistics."""
    model_path = Path(model_dir)
    return {
        "W": np.load(model_path / "W.npy"),
        "b": np.load(model_path / "b.npy"),
        "vocab": np.load(model_path / "vocab.npy", allow_pickle=True).tolist(),
        "bow_mean": np.load(model_path / "bow_mean.npy"),
        "bow_std": np.load(model_path / "bow_std.npy"),
        "num_mean": np.load(model_path / "num_mean.npy"),
        "num_std": np.load(model_path / "num_std.npy"),
    }


def save_artifacts(
    model: LogisticRegression,
    vocab: list[str],
    bow_stats: NormalizationStats,
    num_stats: NormalizationStats,
    model_dir: str | Path,
) -> None:
    """Save model weights and preprocessing artifacts."""
    if model.W is None or model.b is None:
        raise ValueError("Cannot save an untrained model.")

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)

    np.save(model_path / "W.npy", model.W)
    np.save(model_path / "b.npy", model.b)
    np.save(model_path / "vocab.npy", np.array(vocab, dtype=object))
    np.save(model_path / "bow_mean.npy", bow_stats.mean)
    np.save(model_path / "bow_std.npy", bow_stats.std)
    np.save(model_path / "num_mean.npy", num_stats.mean)
    np.save(model_path / "num_std.npy", num_stats.std)
