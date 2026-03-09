"""Project configuration defaults."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PARAPHRASED_DIR = DATA_DIR / "paraphrased"
PROCESSED_DIR = DATA_DIR / "processed"

HUMAN_DATA_PATH = RAW_DIR / "human_texts.csv"
AI_DATA_PATH = RAW_DIR / "ai_texts.csv"
PARAPHRASE_DATA_PATH = PARAPHRASED_DIR / "ai_texts_paraphrased.csv"

DEFAULT_MODEL_PATH = PROCESSED_DIR / "baseline_model.json"
DEFAULT_METRICS_PATH = PROCESSED_DIR / "baseline_metrics.json"
DEFAULT_PREDICTIONS_PATH = PROCESSED_DIR / "baseline_predictions.csv"
DEFAULT_CONFUSION_MATRIX_PATH = PROCESSED_DIR / "baseline_confusion_matrix.svg"
