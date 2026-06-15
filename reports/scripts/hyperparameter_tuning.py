"""Backward-compatible wrapper for hyperparameter tuning experiments."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hybrid_text_classification.hyperparameter_tuning import (  # noqa: E402,F401
    main,
    prepare_data,
    run_all_tuning,
    tune_iterations,
    tune_l2_lambda,
    tune_learning_rates,
    tune_min_frequency,
)


if __name__ == "__main__":
    main()