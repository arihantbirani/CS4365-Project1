"""Experiment pipelines for Week 4-10 deliverables."""

from __future__ import annotations

import json
from typing import Dict, List

import pandas as pd
import numpy as np

from .analysis import (
    build_degradation_summary,
    build_error_analysis,
    build_feature_contribution_rows,
    build_feature_contributions,
    build_final_report,
    build_optimization_summary,
)
from .config import (
    DEFAULT_COMPARISON_PATH,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_CONFUSION_GRID_PATH,
    DEFAULT_DEGRADATION_PATH,
    DEFAULT_DEGRADATION_PLOT_PATH,
    DEFAULT_ERROR_ANALYSIS_PATH,
    DEFAULT_ERROR_CASES_PATH,
    DEFAULT_FEATURE_CONTRIBUTIONS_CSV_PATH,
    DEFAULT_FEATURE_CONTRIBUTIONS_PATH,
    DEFAULT_FEATURE_CONTRIBUTIONS_PLOT_PATH,
    DEFAULT_FINAL_REPORT_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_COMPARISON_PLOT_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OPTIMIZATION_PATH,
    DEFAULT_PREDICTIONS_PATH,
    PARAPHRASE_LEVELS,
    PROCESSED_DIR,
)
from .data import EvaluationSplit, build_experiment_data
from .features import TfidfVectorizer
from .metrics import (
    compute_binary_metrics,
    save_comparison_plot,
    save_confusion_matrix_figure,
    save_confusion_matrix_grid,
    save_degradation_plot,
    save_feature_contribution_plot,
)
from .model import EmbeddingAveragingClassifier, LogisticRegressionGD


def _evaluate_feature_model(
    model_name: str,
    vectorizer: TfidfVectorizer,
    classifier: LogisticRegressionGD,
    train_texts: List[str],
    train_labels: List[int],
    eval_splits: List[EvaluationSplit],
) -> tuple[dict, list[dict], dict]:
    x_train = vectorizer.transform(train_texts)
    classifier.fit(x_train, np.array(train_labels, dtype=float))

    metrics_payload: Dict[str, dict] = {}
    prediction_rows: list[dict] = []

    for split in eval_splits:
        features = vectorizer.transform(split.texts)
        probs = classifier.predict_proba(features)
        preds = classifier.predict(features)
        metrics = compute_binary_metrics(np.array(split.labels), preds).to_dict()
        metrics_payload[split.name] = metrics
        for sample_id, label, pred, prob in zip(split.sample_ids, split.labels, preds.tolist(), probs.tolist()):
            prediction_rows.append(
                {
                    "model": model_name,
                    "split": split.name,
                    "sample_id": sample_id,
                    "true_label": label,
                    "predicted_label": pred,
                    "predicted_probability_ai": round(prob, 4),
                }
            )

    return metrics_payload, prediction_rows, classifier.to_dict()


def _evaluate_neural_model(
    model_name: str,
    classifier: EmbeddingAveragingClassifier,
    train_texts: List[str],
    train_labels: List[int],
    eval_splits: List[EvaluationSplit],
) -> tuple[dict, list[dict], dict]:
    classifier.fit(train_texts, train_labels)

    metrics_payload: Dict[str, dict] = {}
    prediction_rows: list[dict] = []
    for split in eval_splits:
        probs = classifier.predict_proba(split.texts)
        preds = classifier.predict(split.texts)
        metrics = compute_binary_metrics(np.array(split.labels), preds).to_dict()
        metrics_payload[split.name] = metrics
        for sample_id, label, pred, prob in zip(split.sample_ids, split.labels, preds.tolist(), probs.tolist()):
            prediction_rows.append(
                {
                    "model": model_name,
                    "split": split.name,
                    "sample_id": sample_id,
                    "true_label": label,
                    "predicted_label": pred,
                    "predicted_probability_ai": round(prob, 4),
                }
            )

    return metrics_payload, prediction_rows, classifier.to_dict()


def _attach_degradation(metrics_payload: dict) -> dict:
    clean_metrics = metrics_payload["clean"]
    degradation = {}
    for level in PARAPHRASE_LEVELS:
        split_metrics = metrics_payload[level]
        degradation[level] = {
            "accuracy_drop": round(clean_metrics["accuracy"] - split_metrics["accuracy"], 4),
            "f1_drop": round(clean_metrics["f1"] - split_metrics["f1"], 4),
            "recall_drop": round(clean_metrics["recall"] - split_metrics["recall"], 4),
        }
    metrics_payload["degradation"] = degradation
    return metrics_payload


def _run_baseline_optimization(experiment, eval_splits: List[EvaluationSplit]) -> list[dict]:
    trials: list[dict] = []
    for learning_rate, epochs, l2_strength in [
        (0.3, 250, 0.0005),
        (0.5, 400, 0.001),
        (0.8, 500, 0.002),
    ]:
        vectorizer = TfidfVectorizer.fit(experiment.train_texts, min_df=1)
        classifier = LogisticRegressionGD(
            learning_rate=learning_rate,
            epochs=epochs,
            l2_strength=l2_strength,
        )
        metrics_payload, _, _ = _evaluate_feature_model(
            model_name="tfidf_logreg",
            vectorizer=vectorizer,
            classifier=classifier,
            train_texts=experiment.train_texts,
            train_labels=experiment.train_labels,
            eval_splits=eval_splits,
        )
        trials.append(
            {
                "learning_rate": learning_rate,
                "epochs": epochs,
                "l2_strength": l2_strength,
                "clean_f1": metrics_payload["clean"]["f1"],
                "moderate_f1": metrics_payload["moderate"]["f1"],
                "heavy_f1": metrics_payload["heavy"]["f1"],
            }
        )
    return trials


def _run_neural_optimization(experiment, eval_splits: List[EvaluationSplit]) -> list[dict]:
    trials: list[dict] = []
    for embedding_dim, learning_rate, epochs in [
        (16, 0.06, 60),
        (24, 0.08, 80),
        (32, 0.05, 100),
    ]:
        classifier = EmbeddingAveragingClassifier(
            embedding_dim=embedding_dim,
            learning_rate=learning_rate,
            epochs=epochs,
        )
        metrics_payload, _, _ = _evaluate_neural_model(
            model_name="embedding_avg_nn",
            classifier=classifier,
            train_texts=experiment.train_texts,
            train_labels=experiment.train_labels,
            eval_splits=eval_splits,
        )
        trials.append(
            {
                "embedding_dim": embedding_dim,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "clean_f1": metrics_payload["clean"]["f1"],
                "moderate_f1": metrics_payload["moderate"]["f1"],
                "heavy_f1": metrics_payload["heavy"]["f1"],
            }
        )
    return trials


def run_full_experiment() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    experiment = build_experiment_data()
    eval_splits = [experiment.clean_test] + [experiment.paraphrased_tests[level] for level in PARAPHRASE_LEVELS]

    baseline_vectorizer = TfidfVectorizer.fit(experiment.train_texts, min_df=1)
    baseline_classifier = LogisticRegressionGD()
    baseline_metrics, baseline_predictions, baseline_model = _evaluate_feature_model(
        model_name="tfidf_logreg",
        vectorizer=baseline_vectorizer,
        classifier=baseline_classifier,
        train_texts=experiment.train_texts,
        train_labels=experiment.train_labels,
        eval_splits=eval_splits,
    )

    neural_metrics, neural_predictions, neural_model = _evaluate_neural_model(
        model_name="embedding_avg_nn",
        classifier=EmbeddingAveragingClassifier(),
        train_texts=experiment.train_texts,
        train_labels=experiment.train_labels,
        eval_splits=eval_splits,
    )

    baseline_metrics = _attach_degradation(baseline_metrics)
    neural_metrics = _attach_degradation(neural_metrics)

    payload = {
        "dataset_sizes": {
            "train_size": len(experiment.train_texts),
            "clean_test_size": len(experiment.clean_test.texts),
            **{f"{level}_test_size": len(experiment.paraphrased_tests[level].texts) for level in PARAPHRASE_LEVELS},
        },
        "models": {
            "tfidf_logreg": baseline_metrics,
            "embedding_avg_nn": neural_metrics,
        },
    }

    DEFAULT_METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    DEFAULT_MODEL_PATH.write_text(
        json.dumps(
            {
                "tfidf_logreg": baseline_model,
                "embedding_avg_nn": neural_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(baseline_predictions + neural_predictions)
    predictions.to_csv(DEFAULT_PREDICTIONS_PATH, index=False)

    comparison = {
        "clean_accuracy": {
            "tfidf_logreg": baseline_metrics["clean"]["accuracy"],
            "embedding_avg_nn": neural_metrics["clean"]["accuracy"],
        },
        **{
            f"{level}_accuracy": {
                "tfidf_logreg": baseline_metrics[level]["accuracy"],
                "embedding_avg_nn": neural_metrics[level]["accuracy"],
            }
            for level in PARAPHRASE_LEVELS
        },
    }
    DEFAULT_COMPARISON_PATH.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    degradation_summary = build_degradation_summary(payload)
    DEFAULT_DEGRADATION_PATH.write_text(json.dumps(degradation_summary, indent=2), encoding="utf-8")

    error_analysis, error_cases = build_error_analysis(predictions)
    DEFAULT_ERROR_ANALYSIS_PATH.write_text(json.dumps(error_analysis, indent=2), encoding="utf-8")
    error_cases.to_csv(DEFAULT_ERROR_CASES_PATH, index=False)

    save_confusion_matrix_figure(
        baseline_metrics["clean"]["confusion_matrix"],
        title="TF-IDF + Logistic Regression (Clean)",
        output_path=str(DEFAULT_CONFUSION_MATRIX_PATH),
    )
    save_comparison_plot(
        comparison=comparison,
        output_path=str(DEFAULT_MODEL_COMPARISON_PLOT_PATH),
    )
    save_degradation_plot(
        degradation_summary=degradation_summary,
        output_path=str(DEFAULT_DEGRADATION_PLOT_PATH),
    )
    save_confusion_matrix_grid(
        model_metrics=baseline_metrics,
        output_path=str(DEFAULT_CONFUSION_GRID_PATH),
    )

    if baseline_classifier.weights is None:
        raise ValueError("Baseline classifier weights are unavailable after training.")
    feature_contributions = build_feature_contributions(
        baseline_vectorizer.vocabulary_,
        baseline_classifier.weights,
    )
    DEFAULT_FEATURE_CONTRIBUTIONS_PATH.write_text(json.dumps(feature_contributions, indent=2), encoding="utf-8")

    contribution_rows = build_feature_contribution_rows(feature_contributions)
    contribution_rows.to_csv(DEFAULT_FEATURE_CONTRIBUTIONS_CSV_PATH, index=False)
    save_feature_contribution_plot(
        contribution_rows=contribution_rows,
        output_path=str(DEFAULT_FEATURE_CONTRIBUTIONS_PLOT_PATH),
    )

    optimization_summary = build_optimization_summary(
        baseline_search=_run_baseline_optimization(experiment, eval_splits),
        neural_search=_run_neural_optimization(experiment, eval_splits),
    )
    DEFAULT_OPTIMIZATION_PATH.write_text(json.dumps(optimization_summary, indent=2), encoding="utf-8")

    final_report = build_final_report(
        metrics_payload=payload,
        degradation_summary=degradation_summary,
        optimization_summary=optimization_summary,
        feature_contributions=feature_contributions,
        error_analysis=error_analysis,
    )
    DEFAULT_FINAL_REPORT_PATH.write_text(final_report, encoding="utf-8")

    return payload


def run_baseline_experiment() -> dict:
    payload = run_full_experiment()
    baseline_metrics = payload["models"]["tfidf_logreg"]
    clean = baseline_metrics["clean"]
    moderate = baseline_metrics["moderate"]
    return {
        "train_size": payload["dataset_sizes"]["train_size"],
        "clean_test_size": payload["dataset_sizes"]["clean_test_size"],
        "paraphrased_test_size": payload["dataset_sizes"]["moderate_test_size"],
        "clean": clean,
        "paraphrased": moderate,
        "degradation": baseline_metrics["degradation"]["moderate"],
    }
