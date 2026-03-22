"""Models for the detection experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List

import numpy as np

from .features import tokenize


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


@dataclass
class TokenVocabulary:
    token_to_id: Dict[str, int] = field(default_factory=lambda: {"<UNK>": 0})

    @classmethod
    def fit(cls, texts: Iterable[str], min_freq: int = 1) -> "TokenVocabulary":
        counts: Dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1

        vocab = cls()
        for token in sorted(token for token, count in counts.items() if count >= min_freq):
            vocab.token_to_id[token] = len(vocab.token_to_id)
        return vocab

    def encode(self, text: str) -> List[int]:
        ids = [self.token_to_id.get(token, 0) for token in tokenize(text)]
        return ids or [0]

    @property
    def size(self) -> int:
        return len(self.token_to_id)


@dataclass
class EmbeddingAveragingClassifier:
    embedding_dim: int = 24
    learning_rate: float = 0.08
    epochs: int = 80
    l2_strength: float = 0.0005
    seed: int = 7
    vocabulary: TokenVocabulary | None = None
    embeddings: np.ndarray | None = None
    weights: np.ndarray | None = None
    bias: float = 0.0

    def fit(self, texts: List[str], labels: List[int]) -> "EmbeddingAveragingClassifier":
        self.vocabulary = TokenVocabulary.fit(texts)
        encoded = [self.vocabulary.encode(text) for text in texts]
        labels_array = np.array(labels, dtype=float)

        rng = np.random.default_rng(self.seed)
        self.embeddings = rng.normal(0.0, 0.08, size=(self.vocabulary.size, self.embedding_dim))
        self.weights = np.zeros(self.embedding_dim, dtype=float)
        self.bias = 0.0

        for _ in range(self.epochs):
            embedding_grads = np.zeros_like(self.embeddings)
            weight_grad = np.zeros_like(self.weights)
            bias_grad = 0.0

            for ids, label in zip(encoded, labels_array):
                doc_embedding = self.embeddings[ids].mean(axis=0)
                logit = float(doc_embedding @ self.weights + self.bias)
                prob = float(_sigmoid(np.array([logit]))[0])
                error = prob - label

                weight_grad += error * doc_embedding
                bias_grad += error

                token_grad = (error * self.weights) / len(ids)
                for token_id in ids:
                    embedding_grads[token_id] += token_grad

            sample_count = max(1, len(encoded))
            weight_grad = weight_grad / sample_count + self.l2_strength * self.weights
            embedding_grads = embedding_grads / sample_count + self.l2_strength * self.embeddings
            bias_grad = bias_grad / sample_count

            self.weights -= self.learning_rate * weight_grad
            self.embeddings -= self.learning_rate * embedding_grads
            self.bias -= self.learning_rate * bias_grad

        return self

    def _encode_inputs(self, texts: List[str]) -> List[List[int]]:
        if self.vocabulary is None:
            raise ValueError("Model must be fit before prediction.")
        return [self.vocabulary.encode(text) for text in texts]

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if self.embeddings is None or self.weights is None:
            raise ValueError("Model must be fit before prediction.")

        encoded = self._encode_inputs(texts)
        probs = []
        for ids in encoded:
            doc_embedding = self.embeddings[ids].mean(axis=0)
            logit = float(doc_embedding @ self.weights + self.bias)
            probs.append(float(_sigmoid(np.array([logit]))[0]))
        return np.array(probs)

    def predict(self, texts: List[str], threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(texts)
        return (probs >= threshold).astype(int)

    def to_dict(self) -> dict:
        if self.vocabulary is None or self.embeddings is None or self.weights is None:
            raise ValueError("Model must be fit before serialization.")
        return {
            "embedding_dim": self.embedding_dim,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2_strength": self.l2_strength,
            "seed": self.seed,
            "vocabulary_size": self.vocabulary.size,
            "weights": self.weights.tolist(),
            "bias": self.bias,
        }

