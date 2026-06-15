<<<<<<< HEAD
# ArtSense ML

**Hybrid NLP and sentiment-based artwork classification engine**

ArtSense ML predicts which artwork a survey respondent is reacting to by combining free-text responses with structured emotional and behavioral ratings.

## Project Overview

The model classifies responses into one of three paintings:

- *The Persistence of Memory*
- *The Starry Night*
- *The Water Lily Pond*

The pipeline combines:

1. **Text preprocessing** using regex tokenization and Bag-of-Words features
2. **Numeric feature cleaning** for noisy survey responses
3. **Feature normalization** using training-set statistics
4. **Multiclass logistic regression from scratch** using softmax, cross-entropy loss, L2 regularization, and vectorized gradient descent
5. **Saved inference artifacts** for reproducible prediction

## Why This Project Matters

This project demonstrates practical ML engineering skills beyond simply calling a library model:

- implemented logistic regression from scratch with NumPy
- built an end-to-end training and inference pipeline
- handled noisy real-world survey-style data
- combined NLP features with structured numeric features
- saved model artifacts for reuse during inference
- compared model variants including Random Forest, Decision Tree, and Naive Bayes hybrid approaches
- visualized numeric feature distributions and hyperparameter tuning behavior

## Repository Structure

```text
artsense-ml/
├── data/
│   ├── raw/                  # Place raw CSV files here
│   └── processed/            # Optional processed outputs
├── models/
│   └── saved/                # Saved .npy model/preprocessing artifacts
├── notebooks/                # Optional exploratory notebooks
├── src/
│   └── art_sense_ml/
│       ├── artifacts.py      # Save/load model artifacts
│       ├── constants.py      # Label maps and feature column names
│       ├── evaluation.py              # Model comparison experiments
│       ├── features.py                # Hybrid text + numeric feature construction
│       ├── hyperparameter_tuning.py   # Learning-rate/iteration/frequency/L2 sweeps
│       ├── inference.py               # predict_all entry point
│       ├── model.py                   # From-scratch logistic regression
│       ├── preprocessing.py           # CSV cleaning and train/valid/test split
│       ├── text_features.py           # Tokenization, vocabulary, Bag-of-Words
│       ├── train.py                   # Training script
│       └── visualization.py           # Dataset visualization utilities
├── tests/
│   └── test_smoke.py                  # Basic smoke tests
├── pred.py                            # Backward-compatible wrapper
├── data_visualization.py              # Backward-compatible visualization wrapper
├── hyperparameter_tuning.py           # Backward-compatible tuning wrapper
├── model_comparison.py                # Backward-compatible evaluation wrapper
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Installation

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows PowerShell

pip install -r requirements.txt
pip install -e .
```

## Data Setup

Place the original training CSV here:

```text
data/raw/ml_challenge_dataset.csv
```

The raw dataset is not included in this cleaned repository. The saved model artifacts are included under:

```text
models/saved/
```

These artifacts allow prediction to work as long as the input CSV has the same columns as the original challenge dataset.

## Running Predictions

After installing the package:

```bash
python -c "from art_sense_ml import predict_all; print(predict_all('data/raw/ml_challenge_dataset.csv')[:10])"
```

For compatibility with the original course interface, this still works:

```python
from pred import predict_all

predictions = predict_all("data/raw/ml_challenge_dataset.csv")
print(predictions[:10])
```

## Training the Model

```bash
python -m art_sense_ml.train --data data/raw/ml_challenge_dataset.csv --model-dir models/saved
```

Optional hyperparameters:

```bash
python -m art_sense_ml.train \
  --data data/raw/ml_challenge_dataset.csv \
  --model-dir models/saved \
  --lr 0.05 \
  --n-iters 3000 \
  --l2 0.5 \
  --min-freq 5
```

The training script prints train/validation/test loss and accuracy, then saves:

- `W.npy`
- `b.npy`
- `vocab.npy`
- `bow_mean.npy`
- `bow_std.npy`
- `num_mean.npy`
- `num_std.npy`

## Running Model Comparison

```bash
python model_comparison.py --data data/raw/ml_challenge_dataset.csv --runs 10 --output reports/figures/model_comparison.png
```

This compares:

- custom Logistic Regression
- Random Forest
- Random Forest + Naive Bayes probabilities
- Decision Tree

## Creating Visualizations

### Numeric Feature Distributions

To recreate the histogram grid for the numeric survey features:

```bash
python data_visualization.py \
  --data data/raw/ml_challenge_dataset.csv \
  --output reports/figures/data_distribution.png
```

This produces a 2x4 grid showing the distributions of emotional intensity, sentiment ratings, number of colours, number of objects, and willingness to pay.

### Hyperparameter Tuning Plots

To run all hyperparameter experiments and save plots under `reports/figures/`:

```bash
python hyperparameter_tuning.py \
  --data data/raw/ml_challenge_dataset.csv \
  --experiment all \
  --output-dir reports/figures
```

You can also run one experiment at a time:

```bash
python hyperparameter_tuning.py --data data/raw/ml_challenge_dataset.csv --experiment lr
python hyperparameter_tuning.py --data data/raw/ml_challenge_dataset.csv --experiment iters
python hyperparameter_tuning.py --data data/raw/ml_challenge_dataset.csv --experiment minfreq
python hyperparameter_tuning.py --data data/raw/ml_challenge_dataset.csv --experiment lambda
```

The tuning scripts generate:

- `lr_tuning.png`
- `iters_tuning.png`
- `minfreq_tuning.png`
- `lambda_tuning.png`

These plots are useful portfolio artifacts because they show model evaluation and hyperparameter selection rather than just final accuracy.

## Running Tests

```bash
pytest
```

The included smoke tests check that:

- softmax outputs valid probability distributions
- Bag-of-Words feature construction has the expected shape
- the custom logistic regression model can fit a small toy dataset

## Main Technical Details

### Text Features

Text survey responses are concatenated, lowercased, tokenized, and converted into binary Bag-of-Words vectors.

### Numeric Features

The model also uses structured survey features such as emotional intensity, calmness, uneasiness, number of noticed colours, number of noticed objects, and willingness to pay.

### Model

The core classifier is a multiclass logistic regression model implemented with NumPy:

- softmax probability output
- cross-entropy objective
- full-batch gradient descent
- L2 regularization
- vectorized gradients
=======
# Hybrid-NLP-Behavioral-Classification-Engine
Machine learning pipeline combining Bag-of-Words NLP features and structured behavioral survey data to perform multiclass classification using a custom logistic regression implementation.
>>>>>>> 4732d82977756f2305f122e7a919a8e38c4b4d6d
