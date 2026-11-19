from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    age: int = Field(ge=18, le=120)

    tenure_months: int = Field(ge=1, le=120)

    monthly_charges: float = Field(ge=0, le=10000)

    contract: Literal["Month-to-month", "One year", "Two year"]

    internet_service: Literal["DSL", "Fiber optic", "No"]

    support_calls: int = Field(ge=0, le=100)

    paperless_billing: Literal["Yes", "No"]

    payment_method: Literal[
        "Electronic check", "Credit card", "Bank transfer", "Mailed check"
    ]


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    churn_probability: float
    status: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(min_length=1, max_length=100)


class BatchPredictionResponse(BaseModel):
    total_predictions: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_name: str
    framework: str
    task: str
    features: list[str]
    metrics: dict[str, float]
