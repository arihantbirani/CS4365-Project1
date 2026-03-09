"""A lightweight TF-IDF vectorizer implemented with numpy."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


TOKEN_PATTERN = re.compile(r"[a-zA-Z']+")


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


@dataclass
class TfidfVectorizer:
    vocabulary_: Dict[str, int]
    idf_: List[float]

    @classmethod
    def fit(cls, texts: List[str], min_df: int = 1) -> "TfidfVectorizer":
        doc_freq = Counter()
        for text in texts:
            doc_freq.update(set(tokenize(text)))

        terms = sorted(term for term, freq in doc_freq.items() if freq >= min_df)
        vocabulary = {term: idx for idx, term in enumerate(terms)}
        doc_count = len(texts)
        idf = [math.log((1 + doc_count) / (1 + doc_freq[term])) + 1.0 for term in terms]
        return cls(vocabulary_=vocabulary, idf_=idf)

    def transform(self, texts: List[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), len(self.vocabulary_)), dtype=float)
        for row_idx, text in enumerate(texts):
            counts = Counter(tokenize(text))
            total_terms = sum(counts.values()) or 1
            for term, count in counts.items():
                col_idx = self.vocabulary_.get(term)
                if col_idx is None:
                    continue
                tf = count / total_terms
                matrix[row_idx, col_idx] = tf * self.idf_[col_idx]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return matrix / norms

    def to_dict(self) -> dict:
        return {"vocabulary": self.vocabulary_, "idf": self.idf_}

