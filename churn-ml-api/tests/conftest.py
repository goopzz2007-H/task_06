import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import model as ml


@pytest.fixture(scope="session")
def model_file(tmp_path_factory):
    """Train a tiny pipeline once so tests don't depend on the real model file."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, 300),
        "MonthlyCharges": rng.uniform(20, 120, 300),
        "Contract": rng.choice(["Month-to-month", "One year"], 300),
    })
    y = (df.tenure < 24).astype(int).to_numpy()
    pipe = Pipeline([
        ("prep", ColumnTransformer([
            ("num", StandardScaler(), ["tenure", "MonthlyCharges"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Contract"]),
        ])),
        ("clf", LogisticRegression(max_iter=500)),
    ]).fit(df, y)
    path = tmp_path_factory.mktemp("m") / "model.joblib"
    joblib.dump(pipe, path)
    return path


@pytest.fixture()
def client(model_file, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    ml.get_model.cache_clear()
    from app.main import app
    yield TestClient(app)
    ml.get_model.cache_clear()


@pytest.fixture()
def client_no_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    ml.get_model.cache_clear()
    from app.main import app
    yield TestClient(app)
    ml.get_model.cache_clear()


VALID = {"features": {"tenure": 5, "MonthlyCharges": 80.0, "Contract": "Month-to-month"}}
