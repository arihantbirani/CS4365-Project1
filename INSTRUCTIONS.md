# INSTRUCTIONS.md

## Project Summary

This repository contains a CS 4365 project titled **Robust Detection of AI-Generated Text Under Paraphrasing**.

The current implementation covers the early experimental baseline:

- seed datasets for human-written text, AI-generated text, and paraphrased AI-generated text
- a lightweight TF-IDF feature pipeline
- a binary logistic regression classifier implemented in `numpy`
- an evaluation pipeline that compares clean-text performance against paraphrased-text performance
- unit tests for the basic pipeline components

The main research question is whether supervised AI-text detectors remain reliable when AI-generated text is paraphrased.

## Repository Layout

- `README.md`: human-facing project overview and usage
- `docs/project_plan.md`: milestone planning document
- `src/aidetect/`: source code for data loading, feature extraction, model training, metrics, and CLI entry point
- `tests/`: unit tests
- `data/raw/`: human-written and AI-generated source samples
- `data/paraphrased/`: paraphrased AI text samples
- `data/processed/`: generated outputs such as metrics, predictions, and figures

## Environment

Target runtime:

- Python 3.10+ recommended

Current Python dependencies:

- `numpy`
- `pandas`

Install from the repository root with:

```bash
pip install -r requirements.txt
```

## Build / Run / Test

All commands below should be run from the repository root.

### Run the baseline experiment

```bash
PYTHONPATH=src python3 -m aidetect.cli
```

This command will:

- load the datasets from `data/raw/` and `data/paraphrased/`
- train the TF-IDF + logistic regression baseline
- evaluate on a clean test split and a paraphrased test split
- write outputs into `data/processed/`

Expected generated files:

- `data/processed/baseline_metrics.json`
- `data/processed/baseline_model.json`
- `data/processed/baseline_predictions.csv`
- `data/processed/baseline_confusion_matrix.svg`

Additional generated artifact currently present:

- `data/processed/baseline_robustness_summary.svg`

### Run tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Current Code Entry Points

Use these files first when understanding or extending the project:

- `src/aidetect/cli.py`: command-line entry point
- `src/aidetect/pipeline.py`: end-to-end training and evaluation flow
- `src/aidetect/data.py`: dataset loading and train/test split logic
- `src/aidetect/features.py`: TF-IDF implementation
- `src/aidetect/model.py`: logistic regression implementation
- `src/aidetect/metrics.py`: metric computation and SVG figure generation

## Dataset Format

### Human-written data

Path: `data/raw/human_texts.csv`

Columns:

- `sample_id`
- `domain`
- `text`

### AI-generated data

Path: `data/raw/ai_texts.csv`

Columns:

- `sample_id`
- `domain`
- `text`

### Paraphrased AI data

Path: `data/paraphrased/ai_texts_paraphrased.csv`

Columns:

- `sample_id`
- `paraphrase_level`
- `text`

The current seed dataset is intentionally small and is meant to support reproducible baseline development. It is not the final experimental dataset.

## Expected Baseline Behavior

At the current stage, the baseline should produce measurable performance degradation on paraphrased AI text relative to clean AI text. The existing generated metrics show that the paraphrased split is harder for the model than the clean split.

This behavior is expected and aligns with the project hypothesis.

## What Is Implemented vs. Not Yet Implemented

Implemented now:

- repository structure
- seed datasets
- baseline classifier
- evaluation pipeline
- metrics export
- simple visualization outputs
- unit tests

Planned but not yet implemented:

- transformer-based model fine-tuning
- multi-level paraphrasing experiments
- larger real-world datasets
- advanced error analysis
- feature attribution experiments

## Guidance For Another LLM Agent

If you are continuing work on this repository:

1. Start by reading `README.md` and `src/aidetect/pipeline.py`.
2. Run the tests before making changes.
3. Run the baseline experiment and inspect `data/processed/baseline_metrics.json`.
4. Preserve the existing dataset schema unless you intentionally migrate it.
5. Prefer extending the current pipeline rather than replacing it outright, since it serves as the reproducible baseline for later comparisons.
6. If you add stronger models, keep the baseline runnable so the project still supports side-by-side evaluation.

## Recommended Next Steps

- add a second model for comparison, likely a transformer-based classifier
- expand dataset size and domain coverage
- create multiple paraphrase-intensity splits
- add comparison plots for clean vs. paraphrased performance across models
- document experiment settings for reproducibility in the final report
