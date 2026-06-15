"""Text preprocessing and Bag-of-Words feature construction."""

from __future__ import annotations

from collections import Counter
import re

import numpy as np
import pandas as pd

from .constants import TEXT_COLUMNS


def tokenize_lines(data: list[str] | np.ndarray) -> list[list[str]]:
    """Lowercase and tokenize each text line using regex word cleanup."""
    tokenized: list[list[str]] = []
    for line in data:
        cleaned = re.sub(r"\W+", " ", str(line)).lower()
        tokenized.append([word for word in cleaned.split() if word])
    return tokenized


def concatenate_text_columns(df: pd.DataFrame, text_cols: list[str] | None = None) -> list[list[str]]:
    """Concatenate all text response columns and tokenize each row."""
    columns = text_cols or TEXT_COLUMNS
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing expected text columns: {missing}")

    combined = df[columns[0]].fillna("").astype(str)
    for col in columns[1:]:
        combined = combined.str.cat(df[col].fillna("").astype(str), sep=" ")
    return tokenize_lines(combined.to_numpy())


def create_vocab(data: list[list[str]], minimum_frequency: int = 5) -> list[str]:
    """Create a vocabulary containing words that meet the frequency threshold."""
    counts = Counter(word for line in data for word in line)
    return [word for word, count in counts.items() if count >= minimum_frequency]


def make_bow(data: list[list[str]], vocab: list[str]) -> np.ndarray:
    """Create a binary Bag-of-Words matrix with shape [n_samples, n_vocab]."""
    position = {word: idx for idx, word in enumerate(vocab)}
    X = np.zeros((len(data), len(vocab)), dtype=float)

    for row_idx, line in enumerate(data):
        for word in line:
            col_idx = position.get(word)
            if col_idx is not None:
                X[row_idx, col_idx] = 1.0
    return X
