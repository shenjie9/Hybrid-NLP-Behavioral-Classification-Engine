"""Backward-compatible wrapper for dataset visualization."""

from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from art_sense_ml.visualization import main, plot_numeric_feature_distributions  # noqa: E402,F401


if __name__ == "__main__":
    main()
