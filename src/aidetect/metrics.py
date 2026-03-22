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


def save_comparison_plot(comparison: Dict[str, Dict[str, float]], output_path: str) -> None:
    ordered_keys = ["clean_accuracy", "light_accuracy", "moderate_accuracy", "heavy_accuracy"]
    labels = ["Clean", "Light", "Moderate", "Heavy"]
    baseline = [comparison[key]["tfidf_logreg"] for key in ordered_keys]
    neural = [comparison[key]["embedding_avg_nn"] for key in ordered_keys]

    x_positions = [80, 160, 240, 320]
    baseline_points = []
    neural_points = []
    for x_pos, baseline_value, neural_value in zip(x_positions, baseline, neural):
        baseline_points.append((x_pos, 210 - baseline_value * 140))
        neural_points.append((x_pos, 210 - neural_value * 140))

    baseline_line = " ".join(f"{x},{y}" for x, y in baseline_points)
    neural_line = " ".join(f"{x},{y}" for x, y in neural_points)
    labels_svg = []
    for x_pos, label in zip(x_positions, labels):
        labels_svg.append(
            f'<text x="{x_pos}" y="232" text-anchor="middle" font-size="12" fill="#24476b">{label}</text>'
        )

    value_labels = []
    for x_pos, point, value in zip(x_positions, baseline_points, baseline):
        value_labels.append(
            f'<text x="{x_pos}" y="{point[1] - 8}" text-anchor="middle" font-size="11" fill="#2a5c99">{value:.2f}</text>'
        )
    for x_pos, point, value in zip(x_positions, neural_points, neural):
        value_labels.append(
            f'<text x="{x_pos}" y="{point[1] - 8}" text-anchor="middle" font-size="11" fill="#b56b1d">{value:.2f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="420" height="270" viewBox="0 0 420 270">
<rect width="420" height="270" fill="#ffffff" />
<text x="210" y="28" text-anchor="middle" font-size="18" fill="#10263b">Model Accuracy Across Paraphrase Levels</text>
<line x1="50" y1="50" x2="50" y2="210" stroke="#24476b" />
<line x1="50" y1="210" x2="370" y2="210" stroke="#24476b" />
<text x="38" y="214" text-anchor="end" font-size="11" fill="#24476b">0.0</text>
<text x="38" y="144" text-anchor="end" font-size="11" fill="#24476b">0.5</text>
<text x="38" y="74" text-anchor="end" font-size="11" fill="#24476b">1.0</text>
<polyline points="{baseline_line}" fill="none" stroke="#2a5c99" stroke-width="3" />
<polyline points="{neural_line}" fill="none" stroke="#b56b1d" stroke-width="3" />
{''.join(f'<circle cx="{x}" cy="{y}" r="4" fill="#2a5c99" />' for x, y in baseline_points)}
{''.join(f'<circle cx="{x}" cy="{y}" r="4" fill="#b56b1d" />' for x, y in neural_points)}
{''.join(labels_svg)}
{''.join(value_labels)}
<rect x="250" y="42" width="14" height="14" fill="#2a5c99" />
<text x="270" y="54" font-size="12" fill="#24476b">TF-IDF + Logistic Regression</text>
<rect x="250" y="62" width="14" height="14" fill="#b56b1d" />
<text x="270" y="74" font-size="12" fill="#24476b">Embedding Averaging Neural Model</text>
</svg>
"""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(svg)
