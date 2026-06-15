"""Project-wide constants for ArtSense ML."""

from __future__ import annotations

LABEL_TO_ID: dict[str, int] = {
    "The Persistence of Memory": 0,
    "The Starry Night": 1,
    "The Water Lily Pond": 2,
}

ID_TO_LABEL: dict[int, str] = {value: key for key, value in LABEL_TO_ID.items()}

TEXT_COLUMNS: list[str] = [
    "Describe how this painting makes you feel.",
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]

NUMERIC_COLUMNS: list[str] = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?",
]

DEFAULT_MODEL_DIR = "models/saved"
DEFAULT_DATA_PATH = "data/raw/ml_challenge_dataset.csv"
