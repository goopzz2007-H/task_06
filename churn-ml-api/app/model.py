"""Model loading and inference helpers (kept separate from the HTTP layer)."""
import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = "models/champion_churn_model.joblib"
LABELS = {0: "No Churn", 1: "Churn"}


class ModelNotAvailable(RuntimeError):
    pass


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))


@lru_cache(maxsize=1)
def get_model():
    path = model_path()
    if not path.exists():
        raise ModelNotAvailable(f"Model file not found at '{path}'. Train it or set MODEL_PATH.")
    return joblib.load(path)


def model_version() -> str:
    return os.getenv("MODEL_VERSION", "1.0.0")


def expected_columns(model) -> list:
    """Raw input columns the pipeline was trained on (if it exposes them)."""
    cols = getattr(model, "feature_names_in_", None)
    return list(cols) if cols is not None else []


def predict_records(model, records: list) -> list:
    df = pd.DataFrame(records)
    expected = expected_columns(model)
    if expected:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        df = df[expected]  # drop unknown extras, fix column order
    proba = model.predict_proba(df)
    classes = [int(c) for c in model.classes_]
    out = []
    for row in proba:
        probs = {str(c): float(p) for c, p in zip(classes, row)}
        pred = classes[int(row.argmax())]
        out.append({
            "prediction": pred,
            "label": LABELS.get(pred, str(pred)),
            "probability_churn": probs.get("1", 0.0),
            "probabilities": probs,
            "model_version": model_version(),
        })
    return out
