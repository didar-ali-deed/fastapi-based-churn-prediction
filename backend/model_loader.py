import json
from pathlib import Path

import joblib

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


def load_model():
    """Load the trained ML pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    print("ML model loaded successfully!")

    return model


def load_metrics():
    """Load model evaluation metrics."""

    if not METRICS_PATH.exists():
        return {}

    with open(METRICS_PATH, "r") as file:
        metrics = json.load(file)

    return metrics
