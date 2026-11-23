# Customer Churn Prediction: Real-Time ML Inference API

A production-style REST microservice that serves the **champion churn model** (from Task 04) using **FastAPI**, packaged with **Docker** and covered by **pytest** unit tests.

## End-to-end ML system architecture

```
 Raw churn CSV
      |
      v
 [Task 02] Problem framing, responsible data card, risk register
      |
      v
 [Task 03] Leak-free preprocessing: ColumnTransformer (impute + scale + one-hot)
      |
      v
 [Task 04] Train 4+ models, GridSearchCV (stratified K-fold) -> champion Pipeline
      |                                    saved with joblib
      v
 models/champion_churn_model.joblib  (preprocessing + classifier in ONE object)
      |
      v
 [Task 06] FastAPI service  --->  Docker image  --->  any container host
      |
   Client (JSON)  -->  POST /predict  -->  probabilities
```

Because preprocessing lives **inside** the saved sklearn `Pipeline`, the API accepts raw feature values and applies exactly the same transformations as training, which prevents train/serve skew.

## Project structure

```
app/
  main.py      FastAPI routes and error handling
  model.py     model loading (cached) and inference helpers
  schemas.py   Pydantic request/response models
models/        champion_churn_model.joblib
tests/         pytest suite (schema + response codes)
scripts/       train_demo_model.py (demo model generator)
Dockerfile  requirements.txt  requirements-dev.txt
```

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

# Use your real Task 04 model:  copy it to models/champion_churn_model.joblib
# ...or generate a demo model:
python scripts/train_demo_model.py

uvicorn app.main:app --reload
```
Interactive docs: http://localhost:8000/docs

> **Important:** the scikit-learn version in `requirements.txt` must match the version used to train the saved model. Check with `python -c "import sklearn; print(sklearn.__version__)"` in your training environment and update the pin if needed.

## Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## API reference

| Method | Path             | Description                              |
|--------|------------------|------------------------------------------|
| GET    | `/health`        | Liveness and whether the model is loaded |
| GET    | `/model-info`    | Model type, expected features, classes   |
| POST   | `/predict`       | Score one customer                       |
| POST   | `/predict/batch` | Score up to 1000 customers               |

### `POST /predict`

Request (feature names must match your training columns):
```json
{"features": {"tenure": 5, "MonthlyCharges": 80.0, "Contract": "Month-to-month"}}
```
Response `200`:
```json
{
  "prediction": 1,
  "label": "Churn",
  "probability_churn": 0.81,
  "probabilities": {"0": 0.19, "1": 0.81},
  "model_version": "1.0.0"
}
```

| Code | Meaning |
|------|---------|
| 200  | Prediction returned |
| 422  | Invalid payload: missing body fields, missing/invalid feature values |
| 405  | Wrong HTTP method |
| 503  | Model file not found / not loaded |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure": 5, "MonthlyCharges": 80.0, "Contract": "Month-to-month"}}'
```

## Configuration

| Env var         | Default                                | Purpose                    |
|-----------------|----------------------------------------|----------------------------|
| `MODEL_PATH`    | `models/champion_churn_model.joblib`   | Location of the model file |
| `MODEL_VERSION` | `1.0.0`                                | Reported in responses      |

## Tests

```bash
pytest -v
```
The tests train a tiny pipeline on the fly, so they run without the real model. They cover the response schema, probability sums, validation errors (422), method errors (405), missing-model behaviour (503), batch scoring and unknown routes (404).

## Responsible-use notes

Churn scores should support retention outreach, not penalize customers. See the Task 02 data card and risk register for known limitations, and monitor for data drift after deployment.

## Possible improvements

Authentication and rate limiting, request logging and metrics (Prometheus), model registry and versioning, CI pipeline (GitHub Actions) running `pytest` and `docker build`.
