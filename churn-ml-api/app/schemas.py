from typing import Any, Dict, List
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """One customer record: feature name -> value (must match training columns)."""
    features: Dict[str, Any] = Field(
        ...,
        min_length=1,
        description="Raw feature values for a single customer, using the same column names as the training CSV.",
        examples=[{"tenure": 12, "MonthlyCharges": 70.5, "Contract": "Month-to-month"}],
    )


class PredictResponse(BaseModel):
    prediction: int = Field(..., description="Predicted class (1 = churn, 0 = no churn)")
    label: str
    probability_churn: float = Field(..., ge=0, le=1)
    probabilities: Dict[str, float]
    model_version: str


class BatchPredictRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(..., min_length=1, max_length=1000)


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
