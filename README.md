# Robust Detection of AI-Generated Text Under Paraphrasing

**Course:** CS 4365
**Instructor:** Dr. Pu
**Student:** Arihant Birani

## Project Overview

This project evaluates the robustness of supervised AI detection models against paraphrasing attacks. We aim to determine how performance degrades when AI-generated text is paraphrased and whether training augmentation can improve robustness.

## Research Questions

1. How sensitive are supervised classifiers to minor vs. significant paraphrasing?
2. Can we improve detection robustness by including paraphrased samples in the training set?

## Repository Structure

- `src/`: Source code for models and pipelines.
- `data/`: Raw and processed datasets.
- `docs/`: Project documentation and research notes.
- `notebooks/`: EDA and experimental results.
- `tests/`: Unit tests.

## Current Implementation Status

The repository now includes the Week 3-4 deliverables:

- Seed datasets for human-written text, AI-generated text, and paraphrased AI text.
- A reproducible baseline pipeline using TF-IDF features and logistic regression.
- Automatic evaluation on both clean and paraphrased test splits.
- Exported metrics, predictions, and a confusion matrix figure in `data/processed/`.

## Dataset Layout

- `data/raw/human_texts.csv`: Human-written reviews and essays.
- `data/raw/ai_texts.csv`: AI-generated counterparts for the same domains.
- `data/paraphrased/ai_texts_paraphrased.csv`: Paraphrased AI samples used for robustness evaluation.

## Running The Baseline

Run the full experiment from the repository root:

```bash
PYTHONPATH=src python3 -m aidetect.cli
```

This command trains the baseline model and writes:

- `data/processed/baseline_metrics.json`
- `data/processed/baseline_model.json`
- `data/processed/baseline_predictions.csv`
- `data/processed/baseline_confusion_matrix.svg`

## Running Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Baseline Method

The current baseline is intentionally lightweight so the project can run without downloading external ML packages:

1. Training text is vectorized with a custom unigram TF-IDF implementation.
2. A binary logistic regression classifier is trained with batch gradient descent in `numpy`.
3. The model is evaluated on a clean test split and on a paraphrased AI test split.
4. Robustness degradation is reported as the drop in accuracy, recall, and F1 between the two conditions.

This gives you a working experimental foundation for later weeks, where you can swap in stronger models and richer paraphrasing strategies.
