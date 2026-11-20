from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.model_loader import load_metrics, load_model
from backend.predict import predict_customer
from backend.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)


# Application startup and shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    print("Starting ML Deployment API...")

    app.state.model = load_model()
    app.state.metrics = load_metrics()

    yield

    # Shutdown
    print("Shutting down ML Deployment API...")


# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API Using FastAPI",
    description=("A machine learning API powered by Scikit-learn and FastAPI."),
    version="1.0.1",
    lifespan=lifespan,
)


# Root endpoint
@app.get("/")
def home():

    return {
        "message": "Welcome to ML Deployment API",
        "project": "Customer Churn Prediction",
        "version": "1.0.1",
        "docs": "/docs",
        "status": "running",
    }


# Health check
@app.get("/health", response_model=HealthResponse)
def health_check():

    model_loaded = getattr(app.state, "model", None) is not None

    return {
        "status": "healthy" if model_loaded else "unavailable",
        "model_loaded": model_loaded,
    }


# Model information
@app.get("/model/info", response_model=ModelInfoResponse)
def model_info():

    return {
        "model_name": "Random Forest Classifier",
        "framework": "Scikit-learn",
        "task": "Binary Classification",
        "features": [
            "age",
            "tenure_months",
            "monthly_charges",
            "contract",
            "internet_service",
            "support_calls",
            "paperless_billing",
            "payment_method",
        ],
        "metrics": app.state.metrics,
    }


# Single prediction
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):

    try:
        result = predict_customer(app.state.model, customer)

        return result

    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error!s}")


# Batch prediction
@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest):

    try:
        results = []

        for customer in request.customers:
            result = predict_customer(app.state.model, customer)

            results.append(result)

        return {"total_predictions": len(results), "predictions": results}

    except Exception as error:
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {error!s}"
        )
