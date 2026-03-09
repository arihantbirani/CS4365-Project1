"""Experiment pipeline for the Week 4 baseline."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    PROCESSED_DIR,
)
from .data import build_experiment_data
from .features import TfidfVectorizer
from .metrics import compute_binary_metrics, save_confusion_matrix_figure
from .model import LogisticRegressionGD


def run_baseline_experiment() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    experiment = build_experiment_data()

    vectorizer = TfidfVectorizer.fit(experiment.train_texts, min_df=1)
    x_train = vectorizer.transform(experiment.train_texts)
    y_train = np.array(experiment.train_labels, dtype=float)

    model = LogisticRegressionGD()
    model.fit(x_train, y_train)

    x_clean = vectorizer.transform(experiment.clean_test_texts)
    clean_probs = model.predict_proba(x_clean)
    clean_preds = model.predict(x_clean)
    clean_metrics = compute_binary_metrics(np.array(experiment.clean_test_labels), clean_preds)

    x_paraphrased = vectorizer.transform(experiment.paraphrased_test_texts)
    paraphrased_probs = model.predict_proba(x_paraphrased)
    paraphrased_preds = model.predict(x_paraphrased)
    paraphrased_metrics = compute_binary_metrics(
        np.array(experiment.paraphrased_test_labels), paraphrased_preds
    )

    metrics_payload = {
        "train_size": len(experiment.train_texts),
        "clean_test_size": len(experiment.clean_test_texts),
        "paraphrased_test_size": len(experiment.paraphrased_test_texts),
        "clean": clean_metrics.to_dict(),
        "paraphrased": paraphrased_metrics.to_dict(),
        "degradation": {
            "accuracy_drop": round(clean_metrics.accuracy - paraphrased_metrics.accuracy, 4),
            "f1_drop": round(clean_metrics.f1 - paraphrased_metrics.f1, 4),
            "recall_drop": round(clean_metrics.recall - paraphrased_metrics.recall, 4),
        },
    }

    with DEFAULT_METRICS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2)

    with DEFAULT_MODEL_PATH.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "vectorizer": vectorizer.to_dict(),
                "classifier": model.to_dict(),
            },
            handle,
            indent=2,
        )

    predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "split": "clean_test",
                    "sample_id": experiment.clean_test_ids,
                    "true_label": experiment.clean_test_labels,
                    "predicted_label": clean_preds.tolist(),
                    "predicted_probability_ai": clean_probs.round(4),
                }
            ),
            pd.DataFrame(
                {
                    "split": "paraphrased_test",
                    "sample_id": experiment.paraphrased_test_ids,
                    "true_label": experiment.paraphrased_test_labels,
                    "predicted_label": paraphrased_preds.tolist(),
                    "predicted_probability_ai": paraphrased_probs.round(4),
                }
            ),
        ],
        ignore_index=True,
    )
    predictions.to_csv(DEFAULT_PREDICTIONS_PATH, index=False)

    save_confusion_matrix_figure(
        clean_metrics.confusion_matrix,
        title="Clean Test Confusion Matrix",
        output_path=str(DEFAULT_CONFUSION_MATRIX_PATH),
    )

    return metrics_payload
