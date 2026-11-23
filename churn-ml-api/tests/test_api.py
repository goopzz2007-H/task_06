import pytest
from tests.conftest import VALID


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_health_degraded_without_model(client_no_model):
    r = client_no_model.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is False


def test_model_info(client):
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert set(body["expected_features"]) == {"tenure", "MonthlyCharges", "Contract"}
    assert body["classes"] == [0, 1]


def test_predict_success_schema(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "label", "probability_churn", "probabilities", "model_version"}
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_churn"] <= 1.0
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_predict_short_tenure_scores_higher_than_long(client):
    short = client.post("/predict", json={"features": {**VALID["features"], "tenure": 2}}).json()
    long_ = client.post("/predict", json={"features": {**VALID["features"], "tenure": 70}}).json()
    assert short["probability_churn"] > long_["probability_churn"]


def test_predict_ignores_extra_features(client):
    payload = {"features": {**VALID["features"], "unused_field": "x"}}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_missing_feature_returns_422(client):
    r = client.post("/predict", json={"features": {"tenure": 5}})
    assert r.status_code == 422
    assert "Missing required features" in r.json()["detail"]


def test_predict_missing_body_field_returns_422(client):
    assert client.post("/predict", json={}).status_code == 422


def test_predict_empty_features_returns_422(client):
    assert client.post("/predict", json={"features": {}}).status_code == 422


def test_predict_bad_dtype_returns_422(client):
    bad = {"features": {"tenure": "abc", "MonthlyCharges": "xyz", "Contract": "One year"}}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_wrong_method_returns_405(client):
    assert client.get("/predict").status_code == 405


def test_predict_503_when_model_missing(client_no_model):
    assert client_no_model.post("/predict", json=VALID).status_code == 503


def test_batch_predict(client):
    payload = {"records": [VALID["features"], {**VALID["features"], "tenure": 60}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_batch_empty_returns_422(client):
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_unknown_route_returns_404(client):
    assert client.get("/nope").status_code == 404
