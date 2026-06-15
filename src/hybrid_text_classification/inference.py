"""Prediction entry points for ArtSense ML."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import load_artifacts
from .constants import DEFAULT_MODEL_DIR, ID_TO_LABEL
from .features import build_inference_features
from .model import softmax
from .preprocessing import clean_data


def predict_all(filename: str | Path, model_dir: str | Path = DEFAULT_MODEL_DIR) -> list[str]:
    """Predict a painting label for every row in a CSV file.

    This is the single canonical `predict_all` implementation. The root-level
    `pred.py` imports this function for compatibility with the original course
    submission interface.
    """
    df = clean_data(filename)
    artifacts = load_artifacts(model_dir)

    X = build_inference_features(
        df=df,
        vocab=artifacts["vocab"],
        bow_mean=artifacts["bow_mean"],
        bow_std=artifacts["bow_std"],
        num_mean=artifacts["num_mean"],
        num_std=artifacts["num_std"],
    )

    probs = softmax(X @ artifacts["W"].T + artifacts["b"])
    pred_idx = np.argmax(probs, axis=1)
    return [ID_TO_LABEL[int(idx)] for idx in pred_idx]
