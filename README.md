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

The repository now includes the Week 3-10 deliverables:

- Expanded human-written and AI-generated seed datasets for a larger evaluation set.
- A reproducible TF-IDF + logistic regression baseline.
- A stronger comparison model based on trainable averaged token embeddings.
- A deterministic paraphrase generator with `light`, `moderate`, and `heavy` levels.
- Automatic evaluation on clean and multi-level paraphrased test splits.
- Exported metrics, predictions, degradation summaries, confusion matrices, and error-analysis artifacts in `data/processed/`.
- Baseline feature contribution summaries for lexical cue analysis.
- Lightweight optimization sweeps for both implemented models.
- An automatically generated final report artifact for submission support.

## Dataset Layout

- `data/raw/human_texts.csv`: Human-written reviews and essays.
- `data/raw/ai_texts.csv`: AI-generated counterparts for the same domains.
- `data/paraphrased/ai_texts_paraphrased.csv`: Paraphrased AI samples used for robustness evaluation.

## Running The Experiment

Run the full experiment from the repository root:

```bash
PYTHONPATH=src python3 -m aidetect.cli run
```

This command trains both models and writes:

- `data/processed/baseline_metrics.json`
- `data/processed/baseline_model.json`
- `data/processed/baseline_predictions.csv`
- `data/processed/baseline_confusion_matrix.svg`
- `data/processed/model_comparison.json`
- `data/processed/model_comparison.svg`
- `data/processed/degradation_summary.json`
- `data/processed/degradation_summary.svg`
- `data/processed/error_analysis.json`
- `data/processed/error_cases.csv`
- `data/processed/confusion_matrix_grid.svg`
- `data/processed/feature_contributions.json`
- `data/processed/feature_contributions.csv`
- `data/processed/feature_contributions.svg`
- `data/processed/optimization_summary.json`
- `data/processed/final_report.md`

## Generating Paraphrases

To regenerate the paraphrased dataset from the raw AI-written source data:

```bash
PYTHONPATH=src python3 -m aidetect.cli generate-paraphrases
```

## Running Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Method Summary

The current project now supports two models:

1. A TF-IDF + logistic regression baseline.
2. A lightweight neural text model that learns token embeddings and averages them for classification.

The project also includes a rule-based paraphrasing pipeline that produces `light`, `moderate`, and `heavy` paraphrase levels from the AI-written source data. This enables direct robustness comparison across multiple perturbation strengths.

## Week 7-10 Analysis Outputs

The newer analysis layer supports the later milestone tasks from the roadmap:

1. Measuring performance degradation under paraphrasing with explicit per-model drop summaries.
2. Comparing models under different paraphrase strengths while exporting concrete error cases for inspection.
3. Analyzing baseline lexical feature contributions.
4. Running lightweight optimization sweeps for both implemented models.
5. Producing a final Markdown report artifact for the project write-up.

The main analysis artifacts are:

- `data/processed/degradation_summary.json`
- `data/processed/degradation_summary.svg`
- `data/processed/error_analysis.json`
- `data/processed/error_cases.csv`
- `data/processed/confusion_matrix_grid.svg`
- `data/processed/feature_contributions.json`
- `data/processed/feature_contributions.csv`
- `data/processed/feature_contributions.svg`
- `data/processed/optimization_summary.json`
- `data/processed/final_report.md`
