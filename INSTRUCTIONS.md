# INSTRUCTIONS.md

## Project Summary

This repository contains a CS 4365 project titled **Robust Detection of AI-Generated Text Under Paraphrasing**.

The current implementation covers the Week 3-8 experimental foundation:

- expanded seed datasets for human-written text and AI-generated text
- a deterministic paraphrase generator with `light`, `moderate`, and `heavy` levels
- a TF-IDF + logistic regression baseline
- a stronger neural comparison model based on learned token embeddings
- an evaluation pipeline that compares clean-text performance against multiple paraphrase levels
- post-run degradation summaries and error-analysis exports
- unit tests for the core pipeline components

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

### Generate paraphrases

```bash
PYTHONPATH=src python3 -m aidetect.cli generate-paraphrases
```

This command will:

- read the AI-written dataset from `data/raw/ai_texts.csv`
- generate paraphrased versions at `light`, `moderate`, and `heavy` levels
- write the resulting dataset to `data/paraphrased/ai_texts_paraphrased.csv`

### Run the full experiment

```bash
PYTHONPATH=src python3 -m aidetect.cli run
```

This command will:

- load the datasets from `data/raw/` and `data/paraphrased/`
- train both implemented models
- evaluate on a clean test split plus three paraphrase-intensity test splits
- write outputs into `data/processed/`

Expected generated files:

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
- `src/aidetect/paraphrase.py`: paraphrase generation logic
- `src/aidetect/analysis.py`: degradation and error-analysis logic
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

The raw dataset is still a course-project seed dataset rather than a final large benchmark, but it is larger than the initial checkpoint version and now supports broader evaluation.

## Expected Baseline Behavior

At the current stage, the TF-IDF baseline should generally perform strongly on clean text and degrade on at least the stronger paraphrase levels. The neural comparison model provides a second reference point for robustness comparisons.

This behavior is expected and aligns with the project hypothesis.

## What Is Implemented vs. Not Yet Implemented

Implemented now:

- repository structure
- seed datasets
- baseline classifier
- neural comparison classifier
- multi-level paraphrase generation
- evaluation pipeline
- metrics export
- simple visualization outputs
- degradation summaries
- error-case exports
- unit tests

Planned but not yet implemented:

- transformer-based model fine-tuning
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
