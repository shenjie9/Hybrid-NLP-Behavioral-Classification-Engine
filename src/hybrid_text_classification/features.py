"""Feature preparation for text + numeric hybrid ML models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import LABEL_TO_ID, NUMERIC_COLUMNS
from .text_features import concatenate_text_columns, create_vocab, make_bow


@dataclass
class NormalizationStats:
    """Mean and standard deviation used to normalize features."""

    mean: np.ndarray
    std: np.ndarray


def apply_normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize data using training-set mean/std, guarding against zero std."""
    safe_std = np.where(std == 0, 1, std)
    return (X - mean) / safe_std


def fit_normalizer(X: np.ndarray) -> NormalizationStats:
    """Compute normalization statistics for a feature matrix."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std = np.where(std == 0, 1, std)
    return NormalizationStats(mean=mean, std=std)


def ground_truths(df: pd.DataFrame) -> np.ndarray:
    """Map painting labels to integer class IDs."""
    if "Painting" not in df.columns:
        raise KeyError("Expected a 'Painting' column for supervised training/evaluation.")
    return df["Painting"].map(LABEL_TO_ID).astype(int).to_numpy()


def get_numeric_features(df: pd.DataFrame) -> np.ndarray:
    """Extract numeric model features from a cleaned dataframe."""
    missing = [col for col in NUMERIC_COLUMNS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected numeric columns: {missing}")
    return df[NUMERIC_COLUMNS].to_numpy(dtype=float)


def build_training_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    minimum_frequency: int = 5,
) -> dict[str, object]:
    """Build normalized hybrid features for training/validation/testing."""
    train_lines = concatenate_text_columns(train_df)
    valid_lines = concatenate_text_columns(valid_df)
    test_lines = concatenate_text_columns(test_df)

    vocab = create_vocab(train_lines, minimum_frequency=minimum_frequency)

    X_train_bow = make_bow(train_lines, vocab)
    X_valid_bow = make_bow(valid_lines, vocab)
    X_test_bow = make_bow(test_lines, vocab)

    bow_stats = fit_normalizer(X_train_bow)
    X_train_bow_norm = apply_normalize(X_train_bow, bow_stats.mean, bow_stats.std)
    X_valid_bow_norm = apply_normalize(X_valid_bow, bow_stats.mean, bow_stats.std)
    X_test_bow_norm = apply_normalize(X_test_bow, bow_stats.mean, bow_stats.std)

    X_train_num = get_numeric_features(train_df)
    X_valid_num = get_numeric_features(valid_df)
    X_test_num = get_numeric_features(test_df)

    num_stats = fit_normalizer(X_train_num)
    X_train_num_norm = apply_normalize(X_train_num, num_stats.mean, num_stats.std)
    X_valid_num_norm = apply_normalize(X_valid_num, num_stats.mean, num_stats.std)
    X_test_num_norm = apply_normalize(X_test_num, num_stats.mean, num_stats.std)

    return {
        "vocab": vocab,
        "bow_stats": bow_stats,
        "num_stats": num_stats,
        "X_train": np.hstack([X_train_bow_norm, X_train_num_norm]),
        "X_valid": np.hstack([X_valid_bow_norm, X_valid_num_norm]),
        "X_test": np.hstack([X_test_bow_norm, X_test_num_norm]),
        "X_train_bow_raw": X_train_bow,
        "X_valid_bow_raw": X_valid_bow,
        "X_test_bow_raw": X_test_bow,
        "X_train_num_raw": X_train_num,
        "X_valid_num_raw": X_valid_num,
        "X_test_num_raw": X_test_num,
        "t_train": ground_truths(train_df),
        "t_valid": ground_truths(valid_df),
        "t_test": ground_truths(test_df),
    }


def build_inference_features(
    df: pd.DataFrame,
    vocab: list[str],
    bow_mean: np.ndarray,
    bow_std: np.ndarray,
    num_mean: np.ndarray,
    num_std: np.ndarray,
) -> np.ndarray:
    """Build normalized hybrid features for inference using saved artifacts."""
    lines = concatenate_text_columns(df)
    X_text = make_bow(lines, vocab)
    X_num = get_numeric_features(df)

    X_text = apply_normalize(X_text, bow_mean, bow_std)
    X_num = apply_normalize(X_num, num_mean, num_std)
    return np.hstack([X_text, X_num])
