import logging

from fastapi import FastAPI, HTTPException

from app import model as ml
from app.schemas import (BatchPredictRequest, BatchPredictResponse, HealthResponse,
                         PredictRequest, PredictResponse)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("churn-api")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time churn inference served from a scikit-learn champion pipeline.",
    version="1.0.0",
)


def _load_or_503():
    try:
        return ml.get_model()
    except ml.ModelNotAvailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    try:
        ml.get_model()
        return HealthResponse(status="ok", model_loaded=True)
    except ml.ModelNotAvailable:
        return HealthResponse(status="degraded", model_loaded=False)


@app.get("/model-info", tags=["system"])
def model_info():
    model = _load_or_503()
    return {
        "model_version": ml.model_version(),
        "model_type": type(model).__name__,
        "expected_features": ml.expected_columns(model),
        "classes": [int(c) for c in model.classes_],
    }


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest):
    model = _load_or_503()
    try:
        return ml.predict_records(model, [req.features])[0]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:  # bad dtype etc. from the pipeline
        log.exception("Inference failed")
        raise HTTPException(status_code=422, detail=f"Could not score the record: {exc}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
def predict_batch(req: BatchPredictRequest):
    model = _load_or_503()
    try:
        return {"predictions": ml.predict_records(model, req.records)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Batch inference failed")
        raise HTTPException(status_code=422, detail=f"Could not score the records: {exc}")
