"""Deterministic paraphrasing utilities used for robustness evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from .config import AI_DATA_PATH, PARAPHRASE_DATA_PATH, PARAPHRASE_LEVELS


REPLACEMENTS: Dict[str, Dict[str, str]] = {
    "light": {
        "important": "valuable",
        "effective": "useful",
        "students": "learners",
        "improving": "strengthening",
        "community": "local community",
        "provides": "offers",
        "help": "support",
        "improve": "strengthen",
        "shows": "demonstrates",
        "creates": "produces",
    },
    "moderate": {
        "important": "high-value",
        "effective": "successful",
        "students": "students overall",
        "improving": "sharpening",
        "community": "shared community",
        "provides": "makes available",
        "help": "assist",
        "improve": "improve noticeably",
        "shows": "highlights",
        "creates": "builds",
        "because": "since",
        "can": "often can",
    },
    "heavy": {
        "important": "central",
        "effective": "consistently successful",
        "students": "many students",
        "improving": "raising",
        "community": "broader community",
        "provides": "delivers",
        "help": "meaningfully support",
        "improve": "improve in practice",
        "shows": "makes clear",
        "creates": "establishes",
        "because": "largely because",
        "can": "frequently can",
        "strong": "compelling",
        "useful": "practical",
    },
}


LEADERS = {
    "light": "Overall, ",
    "moderate": "In practice, ",
    "heavy": "Viewed more carefully, ",
}


def _replace_terms(text: str, replacements: Dict[str, str]) -> str:
    updated = text
    for source, target in replacements.items():
        updated = updated.replace(f" {source} ", f" {target} ")
        updated = updated.replace(f" {source}.", f" {target}.")
        updated = updated.replace(f" {source},", f" {target},")
        if updated.lower().startswith(source + " "):
            updated = target + updated[len(source) :]
    return updated


def paraphrase_text(text: str, level: str) -> str:
    if level not in PARAPHRASE_LEVELS:
        raise ValueError(f"Unknown paraphrase level: {level}")

    updated = _replace_terms(text, REPLACEMENTS[level])
    if level == "light":
        return LEADERS[level] + updated
    if level == "moderate":
        return LEADERS[level] + updated.replace(" and ", " while also ")
    sentences = [sentence.strip() for sentence in updated.split(".") if sentence.strip()]
    if len(sentences) > 1:
        reordered = ". ".join(reversed(sentences))
    else:
        reordered = updated
    reordered = reordered.replace(" and ", " while ")
    return LEADERS[level] + reordered + ("" if reordered.endswith(".") else ".")


def generate_paraphrase_rows(levels: Iterable[str] = PARAPHRASE_LEVELS) -> pd.DataFrame:
    ai_frame = pd.read_csv(AI_DATA_PATH)
    rows: List[dict] = []
    for _, row in ai_frame.iterrows():
        for level in levels:
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "paraphrase_level": level,
                    "text": paraphrase_text(row["text"], level),
                }
            )
    return pd.DataFrame(rows)


def write_paraphrase_dataset(output_path: Path = PARAPHRASE_DATA_PATH) -> pd.DataFrame:
    frame = generate_paraphrase_rows()
    frame.to_csv(output_path, index=False)
    return frame

