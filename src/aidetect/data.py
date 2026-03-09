"""Data loading and split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from .config import AI_DATA_PATH, HUMAN_DATA_PATH, PARAPHRASE_DATA_PATH


@dataclass
class ExperimentData:
    train_texts: List[str]
    train_labels: List[int]
    clean_test_texts: List[str]
    clean_test_labels: List[int]
    clean_test_ids: List[str]
    paraphrased_test_texts: List[str]
    paraphrased_test_labels: List[int]
    paraphrased_test_ids: List[str]


def load_labeled_frames() -> Dict[str, pd.DataFrame]:
    human = pd.read_csv(HUMAN_DATA_PATH)
    human["label"] = 0

    ai = pd.read_csv(AI_DATA_PATH)
    ai["label"] = 1

    paraphrased = pd.read_csv(PARAPHRASE_DATA_PATH)
    paraphrased["label"] = 1

    return {"human": human, "ai": ai, "paraphrased": paraphrased}


def _split_ids(sample_ids: Iterable[str], test_ratio: float) -> tuple[list[str], list[str]]:
    ordered = sorted(sample_ids)
    test_count = max(1, int(round(len(ordered) * test_ratio)))
    test_ids = ordered[-test_count:]
    train_ids = ordered[:-test_count]
    return train_ids, test_ids


def build_experiment_data(test_ratio: float = 0.25) -> ExperimentData:
    frames = load_labeled_frames()
    human = frames["human"]
    ai = frames["ai"]
    paraphrased = frames["paraphrased"]

    human_train_ids, human_test_ids = _split_ids(human["sample_id"], test_ratio)
    ai_train_ids, ai_test_ids = _split_ids(ai["sample_id"], test_ratio)

    human_train = human[human["sample_id"].isin(human_train_ids)]
    ai_train = ai[ai["sample_id"].isin(ai_train_ids)]
    human_test = human[human["sample_id"].isin(human_test_ids)]
    ai_test = ai[ai["sample_id"].isin(ai_test_ids)]
    paraphrased_test = paraphrased[paraphrased["sample_id"].isin(ai_test_ids)]

    train = pd.concat([human_train, ai_train], ignore_index=True)
    clean_test = pd.concat([human_test, ai_test], ignore_index=True)
    paraphrased_eval = pd.concat([human_test, paraphrased_test], ignore_index=True)

    return ExperimentData(
        train_texts=train["text"].tolist(),
        train_labels=train["label"].tolist(),
        clean_test_texts=clean_test["text"].tolist(),
        clean_test_labels=clean_test["label"].tolist(),
        clean_test_ids=clean_test["sample_id"].tolist(),
        paraphrased_test_texts=paraphrased_eval["text"].tolist(),
        paraphrased_test_labels=paraphrased_eval["label"].tolist(),
        paraphrased_test_ids=paraphrased_eval["sample_id"].tolist(),
    )

