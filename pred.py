"""Backward-compatible entry point for the original CSC311 submission interface.

The project has been refactored into the `src/art_sense_ml/` package. This file
keeps old imports working, especially `from pred import predict_all`.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from art_sense_ml.features import apply_normalize, ground_truths  # noqa: E402
from art_sense_ml.inference import predict_all  # noqa: E402
from art_sense_ml.model import LogisticRegression, accuracy, cross_entropy_loss, softmax  # noqa: E402
from art_sense_ml.preprocessing import clean_data, extract_number, split_data  # noqa: E402
from art_sense_ml.text_features import (  # noqa: E402
    concatenate_text_columns as concatenate_string_df,
    create_vocab,
    make_bow,
    tokenize_lines as reformat_strings,
)
from art_sense_ml.train import train as pred  # noqa: E402


if __name__ == "__main__":
    default_csv = Path("data/raw/ml_challenge_dataset.csv")
    if default_csv.exists():
        preds = predict_all(default_csv)
        print(preds[:10])
    else:
        print(
            "No dataset found at data/raw/ml_challenge_dataset.csv. "
            "Place the CSV there or call predict_all('path/to/file.csv')."
        )
