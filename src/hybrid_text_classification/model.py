"""From-scratch multiclass logistic regression model."""

from __future__ import annotations

import numpy as np


def softmax(X: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = X - np.max(X, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    return numerator / denominator


def cross_entropy_loss(probs: np.ndarray, t: np.ndarray) -> float:
    """Mean cross-entropy loss for integer class labels."""
    n = len(t)
    clipped = np.clip(probs[np.arange(n), t], 1e-12, 1.0)
    return float(-np.mean(np.log(clipped)))


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Classification accuracy."""
    return float(np.mean(y_pred == y_true))


class LogisticRegression:
    """Multiclass logistic regression trained with vectorized gradient descent."""

    def __init__(
        self,
        lr: float = 0.05,
        n_iters: int = 3000,
        n_classes: int = 3,
        l2_lambda: float = 0.5,
        random_state: int | None = 42,
        tolerance: float = 1e-4,
    ) -> None:
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.lr = lr
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.l2_lambda = l2_lambda
        self.random_state = random_state
        self.tolerance = tolerance

    def fit(self, X: np.ndarray, t: np.ndarray) -> None:
        """Train the model using full-batch gradient descent."""
        rng = np.random.default_rng(self.random_state)
        _, n_features = X.shape
        self.W = rng.normal(loc=0.0, scale=0.01, size=(self.n_classes, n_features))
        self.b = np.zeros(self.n_classes)

        prev_loss = float("inf")
        for _ in range(self.n_iters):
            probs = self.predict_proba(X)
            curr_loss = cross_entropy_loss(probs, t)
            if abs(prev_loss - curr_loss) < self.tolerance:
                break
            prev_loss = curr_loss

            dW, db = self.gradient(X, t, probs)
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def gradient(self, X: np.ndarray, t: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute gradients for weights and bias."""
        if self.W is None:
            raise ValueError("Model weights are not initialized. Call fit() first.")

        n_samples = X.shape[0]
        Y = np.zeros((n_samples, self.n_classes))
        Y[np.arange(n_samples), t] = 1

        dW = ((probs - Y).T @ X) / n_samples + self.l2_lambda * self.W
        db = np.mean(probs - Y, axis=0)
        return dW, db

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if self.W is None or self.b is None:
            raise ValueError("Model weights are not initialized. Call fit() or load artifacts first.")
        return softmax(X @ self.W.T + self.b)

    # Backward-compatible alias used by the original project.
    predict = predict_proba

    def predict_class(self, X: np.ndarray) -> np.ndarray:
        """Predict integer class labels."""
        return np.argmax(self.predict_proba(X), axis=1)
