"""Data cleaning utilities for noisy artwork response survey data."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract_number(value: object) -> float:
    """Extract the first numeric value from a possibly messy survey response."""
    match = re.search(r"-?\d+(\.\d+)?", str(value))
    if match:
        return float(match.group())
    return np.nan


def clean_data(path: str | Path) -> pd.DataFrame:
    """Load and clean the raw CSV used by the artwork classifier.

    The original dataset contains free-form survey responses, including values
    like "$50", "around 7", blanks, and occasional outliers. This function
    converts these into stable numeric columns for model training/inference.
    """
    df = pd.read_csv(path)

    # Clean column 2: coerce to rounded integer and fill missing with median.
    col_name = df.columns[2]
    df[col_name] = pd.to_numeric(df[col_name], errors="coerce").round()
    df[col_name] = df[col_name].fillna(df[col_name].median()).astype(int)

    # Clean columns 4-7: extract first number from messy text.
    for col in df.columns[4:8]:
        df[col] = df[col].apply(extract_number)
        df[col] = df[col].fillna(df[col].median()).astype(int)

    # Clean columns 8-9: replace outliers/zeros with median.
    for col in df.columns[8:10]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        is_outlier = (df[col] < lower_bound) | (df[col] > upper_bound)
        is_zero = df[col] == 0
        df.loc[is_outlier | is_zero, col] = np.nan
        df[col] = df[col].fillna(df[col].median()).astype(int)

    # Clean column 10: extract numbers, then replace zeros/outliers with median.
    col = df.columns[10]
    df[col] = df[col].apply(extract_number)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df.loc[(df[col] == 0) | (df[col] > upper_bound) | (df[col] < lower_bound), col] = np.nan
    df[col] = df[col].fillna(df[col].median()).astype(int)

    return df


def split_data(
    df: pd.DataFrame,
    train_end: float = 0.70,
    valid_end: float = 0.85,
    random_state: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a dataframe into train, validation, and test sets."""
    if not 0 < train_end < valid_end < 1:
        raise ValueError("Expected 0 < train_end < valid_end < 1.")

    shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    first_split = int(train_end * len(shuffled))
    second_split = int(valid_end * len(shuffled))

    train_df = shuffled.iloc[:first_split]
    valid_df = shuffled.iloc[first_split:second_split]
    test_df = shuffled.iloc[second_split:]
    return train_df, valid_df, test_df
