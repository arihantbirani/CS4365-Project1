"""Evaluation helpers for binary classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: List[List[int]]

    def to_dict(self) -> Dict[str, float | List[List[int]]]:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "confusion_matrix": self.confusion_matrix,
        }


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    accuracy = (tp + tn) / max(1, len(y_true))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=[[tn, fp], [fn, tp]],
    )


def save_confusion_matrix_figure(matrix: List[List[int]], title: str, output_path: str) -> None:
    cells = []
    colors = ["#dce8f5", "#8fb6df", "#8fb6df", "#4c84c3"]
    values = [matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]]
    coordinates = [(70, 80), (190, 80), (70, 200), (190, 200)]

    for idx, ((x_pos, y_pos), value) in enumerate(zip(coordinates, values)):
        cells.append(
            f'<rect x="{x_pos}" y="{y_pos}" width="110" height="110" fill="{colors[idx]}" stroke="#24476b" />'
        )
        cells.append(
            f'<text x="{x_pos + 55}" y="{y_pos + 62}" text-anchor="middle" font-size="24" fill="#10263b">{value}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="360" height="360" viewBox="0 0 360 360">
<rect width="360" height="360" fill="#ffffff" />
<text x="180" y="32" text-anchor="middle" font-size="18" fill="#10263b">{title}</text>
<text x="125" y="70" text-anchor="middle" font-size="14" fill="#24476b">Pred: Human</text>
<text x="245" y="70" text-anchor="middle" font-size="14" fill="#24476b">Pred: AI</text>
<text x="25" y="145" text-anchor="middle" font-size="14" fill="#24476b" transform="rotate(-90 25 145)">True: Human</text>
<text x="25" y="265" text-anchor="middle" font-size="14" fill="#24476b" transform="rotate(-90 25 265)">True: AI</text>
{''.join(cells)}
</svg>
"""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(svg)
