import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = Path("data/processed/ml_challenge_cleaned_dataset.csv")
OUTPUT_PATH = Path("reports/figures/numeric_feature_distributions.png")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

numeric_cols = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?",
]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

titles = [
    "Emotion Intensity",
    "Feel Sombre",
    "Feel Content",
    "Feel Calm",
    "Feel Uneasy",
    "Num Colours",
    "Num Objects",
    "Willingness to Pay",
]

for ax, col, title in zip(axes, numeric_cols, titles):
    ax.hist(df[col].dropna(), bins=20)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved figure to {OUTPUT_PATH}")