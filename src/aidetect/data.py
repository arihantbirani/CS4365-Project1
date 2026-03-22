"""Data loading and split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

from .config import AI_DATA_PATH, HUMAN_DATA_PATH, PARAPHRASE_DATA_PATH, PARAPHRASE_LEVELS


@dataclass
class EvaluationSplit:
    name: str
    texts: List[str]
    labels: List[int]
    sample_ids: List[str]


@dataclass
class ExperimentData:
    train_texts: List[str]
    train_labels: List[int]
    clean_test: EvaluationSplit
    paraphrased_tests: Dict[str, EvaluationSplit]


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
    test_count = max(2, int(round(len(ordered) * test_ratio)))
    test_ids = ordered[-test_count:]
    train_ids = ordered[:-test_count]
    return train_ids, test_ids


def _build_split(name: str, frame: pd.DataFrame) -> EvaluationSplit:
    return EvaluationSplit(
        name=name,
        texts=frame["text"].tolist(),
        labels=frame["label"].tolist(),
        sample_ids=frame["sample_id"].tolist(),
    )


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

    train = pd.concat([human_train, ai_train], ignore_index=True)
    clean_test = pd.concat([human_test, ai_test], ignore_index=True)

    paraphrased_tests: Dict[str, EvaluationSplit] = {}
    for level in PARAPHRASE_LEVELS:
        paraphrased_level = paraphrased[
            (paraphrased["sample_id"].isin(ai_test_ids)) & (paraphrased["paraphrase_level"] == level)
        ]
        paraphrased_eval = pd.concat([human_test, paraphrased_level], ignore_index=True)
        paraphrased_tests[level] = _build_split(name=level, frame=paraphrased_eval)

    return ExperimentData(
        train_texts=train["text"].tolist(),
        train_labels=train["label"].tolist(),
        clean_test=_build_split(name="clean", frame=clean_test),
        paraphrased_tests=paraphrased_tests,
    )

