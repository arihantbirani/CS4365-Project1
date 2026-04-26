"""Post-evaluation analysis for robustness, feature behavior, and reporting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from .config import PARAPHRASE_LEVELS
from .data import build_experiment_data


def build_degradation_summary(metrics_payload: dict) -> dict:
    summary = {}
    for model_name, model_metrics in metrics_payload["models"].items():
        clean = model_metrics["clean"]
        summary[model_name] = {
            "clean_accuracy": clean["accuracy"],
            "clean_f1": clean["f1"],
            "levels": {},
        }
        for level in PARAPHRASE_LEVELS:
            level_metrics = model_metrics[level]
            summary[model_name]["levels"][level] = {
                "accuracy": level_metrics["accuracy"],
                "f1": level_metrics["f1"],
                "recall": level_metrics["recall"],
                "accuracy_drop": model_metrics["degradation"][level]["accuracy_drop"],
                "f1_drop": model_metrics["degradation"][level]["f1_drop"],
                "recall_drop": model_metrics["degradation"][level]["recall_drop"],
            }
    return summary


def build_error_analysis(predictions: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    experiment = build_experiment_data()
    source_texts: Dict[str, str] = {}
    for split in [experiment.clean_test] + list(experiment.paraphrased_tests.values()):
        for sample_id, text in zip(split.sample_ids, split.texts):
            source_texts[f"{split.name}:{sample_id}"] = text

    clean_predictions = predictions[predictions["split"] == "clean"][
        ["model", "sample_id", "predicted_label", "predicted_probability_ai", "true_label"]
    ].rename(
        columns={
            "predicted_label": "clean_predicted_label",
            "predicted_probability_ai": "clean_probability_ai",
        }
    )

    paraphrased_predictions = predictions[predictions["split"] != "clean"].copy()
    joined = paraphrased_predictions.merge(clean_predictions, on=["model", "sample_id", "true_label"], how="left")

    joined["flip_from_clean"] = joined["predicted_label"] != joined["clean_predicted_label"]
    joined["new_error_under_paraphrase"] = (
        (joined["clean_predicted_label"] == joined["true_label"])
        & (joined["predicted_label"] != joined["true_label"])
    )

    joined["text"] = joined.apply(lambda row: source_texts.get(f"{row['split']}:{row['sample_id']}", ""), axis=1)
    joined["clean_text"] = joined.apply(
        lambda row: source_texts.get(f"clean:{row['sample_id']}", ""),
        axis=1,
    )

    ai_rows = joined[joined["true_label"] == 1]
    human_rows = joined[joined["true_label"] == 0]

    analysis = {
        "summary": {},
        "notable_cases": {},
    }

    for model_name, model_rows in joined.groupby("model"):
        analysis["summary"][model_name] = {
            "total_paraphrased_rows": int(len(model_rows)),
            "flip_count": int(model_rows["flip_from_clean"].sum()),
            "new_error_count": int(model_rows["new_error_under_paraphrase"].sum()),
            "ai_false_negative_count": int(
                ((model_rows["true_label"] == 1) & (model_rows["predicted_label"] == 0)).sum()
            ),
            "human_false_positive_count": int(
                ((model_rows["true_label"] == 0) & (model_rows["predicted_label"] == 1)).sum()
            ),
            "per_level": {},
        }

        for level in PARAPHRASE_LEVELS:
            level_rows = model_rows[model_rows["split"] == level]
            analysis["summary"][model_name]["per_level"][level] = {
                "flip_count": int(level_rows["flip_from_clean"].sum()),
                "new_error_count": int(level_rows["new_error_under_paraphrase"].sum()),
                "ai_false_negatives": int(
                    ((level_rows["true_label"] == 1) & (level_rows["predicted_label"] == 0)).sum()
                ),
                "human_false_positives": int(
                    ((level_rows["true_label"] == 0) & (level_rows["predicted_label"] == 1)).sum()
                ),
            }

        false_negative_cases = ai_rows[
            (ai_rows["model"] == model_name) & (ai_rows["predicted_label"] == 0)
        ].sort_values(["split", "predicted_probability_ai"])
        false_positive_cases = human_rows[
            (human_rows["model"] == model_name) & (human_rows["predicted_label"] == 1)
        ].sort_values(["split", "predicted_probability_ai"], ascending=[True, False])

        analysis["notable_cases"][model_name] = {
            "ai_false_negatives": false_negative_cases[
                ["split", "sample_id", "predicted_probability_ai", "clean_probability_ai"]
            ]
            .head(5)
            .to_dict(orient="records"),
            "human_false_positives": false_positive_cases[
                ["split", "sample_id", "predicted_probability_ai", "clean_probability_ai"]
            ]
            .head(5)
            .to_dict(orient="records"),
        }

    error_cases = joined[joined["predicted_label"] != joined["true_label"]][
        [
            "model",
            "split",
            "sample_id",
            "true_label",
            "predicted_label",
            "predicted_probability_ai",
            "clean_predicted_label",
            "clean_probability_ai",
            "flip_from_clean",
            "new_error_under_paraphrase",
            "clean_text",
            "text",
        ]
    ].sort_values(["model", "split", "sample_id"])

    return analysis, error_cases


def build_feature_contributions(vocabulary: dict[str, int], weights: np.ndarray, top_k: int = 15) -> dict:
    """Summarize the strongest lexical cues learned by the baseline model."""

    sorted_terms = sorted(vocabulary.items(), key=lambda item: item[1])
    term_weights = [(term, float(weights[index])) for term, index in sorted_terms]
    positive = sorted(term_weights, key=lambda item: item[1], reverse=True)[:top_k]
    negative = sorted(term_weights, key=lambda item: item[1])[:top_k]

    return {
        "top_ai_indicators": [
            {"term": term, "weight": round(weight, 4)}
            for term, weight in positive
        ],
        "top_human_indicators": [
            {"term": term, "weight": round(weight, 4)}
            for term, weight in negative
        ],
    }


def build_feature_contribution_rows(contributions: dict) -> pd.DataFrame:
    rows: list[dict] = []
    for direction, items in contributions.items():
        label = "ai_indicator" if direction == "top_ai_indicators" else "human_indicator"
        for rank, item in enumerate(items, start=1):
            rows.append(
                {
                    "direction": label,
                    "rank": rank,
                    "term": item["term"],
                    "weight": item["weight"],
                }
            )
    return pd.DataFrame(rows)


def build_optimization_summary(
    baseline_search: list[dict],
    neural_search: list[dict],
) -> dict:
    """Package Week 9 optimization experiments into one summary object."""

    best_baseline = max(baseline_search, key=lambda row: row["moderate_f1"])
    best_neural = max(neural_search, key=lambda row: row["moderate_f1"])
    return {
        "baseline_logreg": {
            "best": best_baseline,
            "trials": baseline_search,
        },
        "embedding_avg_nn": {
            "best": best_neural,
            "trials": neural_search,
        },
    }


def build_final_report(
    metrics_payload: dict,
    degradation_summary: dict,
    optimization_summary: dict,
    feature_contributions: dict,
    error_analysis: dict,
) -> str:
    """Generate a concise final project report artifact for Week 10."""

    dataset_sizes = metrics_payload["dataset_sizes"]
    baseline_clean = metrics_payload["models"]["tfidf_logreg"]["clean"]
    neural_clean = metrics_payload["models"]["embedding_avg_nn"]["clean"]
    baseline_heavy = metrics_payload["models"]["tfidf_logreg"]["heavy"]
    neural_heavy = metrics_payload["models"]["embedding_avg_nn"]["heavy"]

    lines = [
        "# Final Report: Robust Detection of AI-Generated Text Under Paraphrasing",
        "",
        "## Objective",
        "Measure how strongly supervised AI-text detectors degrade under paraphrasing and whether simple model choices remain robust.",
        "",
        "## Dataset Summary",
        f"- Train size: {dataset_sizes['train_size']}",
        f"- Clean test size: {dataset_sizes['clean_test_size']}",
        f"- Light paraphrase test size: {dataset_sizes['light_test_size']}",
        f"- Moderate paraphrase test size: {dataset_sizes['moderate_test_size']}",
        f"- Heavy paraphrase test size: {dataset_sizes['heavy_test_size']}",
        "",
        "## Model Results",
        f"- TF-IDF + Logistic Regression clean accuracy/F1: {baseline_clean['accuracy']:.4f} / {baseline_clean['f1']:.4f}",
        f"- Embedding Averaging Neural Model clean accuracy/F1: {neural_clean['accuracy']:.4f} / {neural_clean['f1']:.4f}",
        f"- TF-IDF + Logistic Regression heavy paraphrase accuracy/F1: {baseline_heavy['accuracy']:.4f} / {baseline_heavy['f1']:.4f}",
        f"- Embedding Averaging Neural Model heavy paraphrase accuracy/F1: {neural_heavy['accuracy']:.4f} / {neural_heavy['f1']:.4f}",
        "",
        "## Robustness Findings",
    ]

    for model_name, summary in degradation_summary.items():
        heavy = summary["levels"]["heavy"]
        lines.append(
            f"- {model_name}: heavy paraphrase accuracy drop {heavy['accuracy_drop']:.4f}, F1 drop {heavy['f1_drop']:.4f}"
        )

    lines.extend(
        [
            "",
            "## Feature Contribution Analysis",
            "- Strong AI-indicator terms:",
        ]
    )
    lines.extend(
        [f"  - {item['term']} ({item['weight']:.4f})" for item in feature_contributions["top_ai_indicators"][:5]]
    )
    lines.append("- Strong human-indicator terms:")
    lines.extend(
        [f"  - {item['term']} ({item['weight']:.4f})" for item in feature_contributions["top_human_indicators"][:5]]
    )

    lines.extend(
        [
            "",
            "## Optimization Summary",
            f"- Best baseline moderate-F1: {optimization_summary['baseline_logreg']['best']['moderate_f1']:.4f}",
            f"- Best neural moderate-F1: {optimization_summary['embedding_avg_nn']['best']['moderate_f1']:.4f}",
            "",
            "## Error Analysis",
        ]
    )

    for model_name, summary in error_analysis["summary"].items():
        lines.append(
            f"- {model_name}: {summary['flip_count']} prediction flips from clean to paraphrased text, {summary['new_error_count']} new paraphrase-induced errors."
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "Both models remain strong on clean text, but paraphrasing reduces reliability. Lexical features still capture useful cues, yet robustness under heavier paraphrasing remains an open improvement target.",
            "",
        ]
    )
    return "\n".join(lines)
