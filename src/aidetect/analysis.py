"""Post-evaluation analysis for robustness and error behavior."""

from __future__ import annotations

from typing import Dict, List

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

