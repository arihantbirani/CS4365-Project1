"""Simple logistic regression trained with batch gradient descent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class LogisticRegressionGD:
    learning_rate: float = 0.5
    epochs: int = 400
    l2_strength: float = 0.001
    weights: np.ndarray | None = None
    bias: float = 0.0

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "LogisticRegressionGD":
        sample_count, feature_count = features.shape
        self.weights = np.zeros(feature_count, dtype=float)
        self.bias = 0.0

        for _ in range(self.epochs):
            logits = features @ self.weights + self.bias
            probs = _sigmoid(logits)
            errors = probs - labels

            weight_grad = (features.T @ errors) / sample_count
            weight_grad += self.l2_strength * self.weights
            bias_grad = errors.mean()

            self.weights -= self.learning_rate * weight_grad
            self.bias -= self.learning_rate * bias_grad

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model must be fit before prediction.")
        logits = features @ self.weights + self.bias
        return _sigmoid(logits)

    def predict(self, features: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(features)
        return (probs >= threshold).astype(int)

    def to_dict(self) -> dict:
        if self.weights is None:
            raise ValueError("Model must be fit before serialization.")
        return {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2_strength": self.l2_strength,
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }

