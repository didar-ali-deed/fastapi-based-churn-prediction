import pytest
from fastapi.testclient import TestClient

from backend.main import app


@pytest.fixture(scope="module")
def client():

    with TestClient(app) as test_client:
        yield test_client


# Valid customer input
VALID_CUSTOMER = {
    "age": 28,
    "tenure_months": 12,
    "monthly_charges": 75.5,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "support_calls": 3,
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
}


def test_root_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_model_info(client):

    response = client.get("/model/info")

    assert response.status_code == 200

    data = response.json()

    assert data["framework"] == "Scikit-learn"
    assert data["task"] == "Binary Classification"
    assert len(data["features"]) == 8


def test_single_prediction(client):

    response = client.post("/predict", json=VALID_CUSTOMER)

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]

    assert result["label"] in ["Churn", "No Churn"]

    assert 0 <= result["churn_probability"] <= 1

    assert result["status"] == "success"


def test_batch_prediction(client):

    response = client.post(
        "/predict/batch",
        json={
            "customers": [
                VALID_CUSTOMER,
                {
                    **VALID_CUSTOMER,
                    "age": 55,
                    "tenure_months": 60,
                    "contract": "Two year",
                },
            ]
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["total_predictions"] == 2
    assert len(data["predictions"]) == 2


def test_invalid_age(client):

    invalid_customer = {**VALID_CUSTOMER, "age": -5}

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422


def test_invalid_contract(client):

    invalid_customer = {**VALID_CUSTOMER, "contract": "Lifetime"}

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422


def test_missing_field(client):

    invalid_customer = VALID_CUSTOMER.copy()

    del invalid_customer["age"]

    response = client.post("/predict", json=invalid_customer)

    assert response.status_code == 422
