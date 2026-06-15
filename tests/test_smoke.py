from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from art_sense_ml.model import LogisticRegression, softmax
from art_sense_ml.text_features import create_vocab, make_bow, tokenize_lines


def test_softmax_rows_sum_to_one():
    X = np.array([[1.0, 2.0, 3.0], [1000.0, 1001.0, 1002.0]])
    probs = softmax(X)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))


def test_bow_feature_shape():
    lines = tokenize_lines(["calm blue blue", "uneasy bright red"])
    vocab = create_vocab(lines, minimum_frequency=1)
    X = make_bow(lines, vocab)
    assert X.shape == (2, len(vocab))


def test_logistic_regression_can_fit_small_dataset():
    X = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
        [0.1, 0.9],
        [0.5, 0.5],
        [0.45, 0.55],
    ])
    y = np.array([0, 0, 1, 1, 2, 2])
    model = LogisticRegression(lr=0.1, n_iters=200, n_classes=3, l2_lambda=0.0)
    model.fit(X, y)
    preds = model.predict_class(X)
    assert preds.shape == y.shape
