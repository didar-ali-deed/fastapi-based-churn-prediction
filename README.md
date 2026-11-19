# FastAPI-Based Customer Churn Prediction

A machine learning project that predicts customer churn using Python and Scikit-learn, with a FastAPI backend for serving predictions and a Streamlit frontend for user interaction.

The project includes model training, data generation, prediction APIs, input validation, and automated API testing.

## Project Structure

```text
ml-deployment-project/
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── model_loader.py
│   ├── predict.py
│   └── schemas.py
│
├── data/
│   └── customer_churn.csv
│
├── frontend/
│   └── app.py
│
├── models/
│   ├── churn_pipeline.joblib
│   └── metrics.json
│
├── training/
│   ├── generate_data.py
│   └── train_model.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── pytest.ini
└── requirements.txt
```

## Technologies Used

- **Python** – Core programming language
- **Pandas & NumPy** – Data manipulation and preprocessing
- **Scikit-learn** – Machine learning and model training
- **Joblib** – Model serialization
- **FastAPI** – REST API development
- **Pydantic** – Request data validation
- **Streamlit** – Interactive frontend
- **Pytest** – Automated testing

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/didar-ali-deed/fastapi-based-churn-prediction.git
cd fastapi-based-churn-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### 1. Start the FastAPI Backend

```bash
uvicorn backend.main:app --reload
```

API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### 2. Start the Streamlit Frontend

Open another terminal and activate the virtual environment.

```bash
streamlit run frontend/app.py
```

### 3. Train the Machine Learning Model

To generate training data:

```bash
python training/generate_data.py
```

To train the model:

```bash
python training/train_model.py
```

## Running Tests

Run the automated API tests using:

```bash
python -m pytest -v
```

### Test Results

The project currently includes 8 automated API tests.

- Root endpoint
- Health endpoint
- Model information
- Single prediction
- Batch prediction
- Invalid age validation
- Invalid contract validation
- Missing field validation

**Latest test results: 8 passed.**

## Project Features

- Machine learning-based customer churn prediction
- RESTful API using FastAPI
- Single and batch prediction endpoints
- Request validation using Pydantic
- Serialized model loading using Joblib
- Interactive Streamlit interface
- Automated API testing using Pytest

## Author

**Didar Ali**

- GitHub: [didar-ali-deed](https://github.com/didar-ali-deed)
- LinkedIn: [Didar Ali](https://www.linkedin.com/in/didar-ali-deed/)

---

*Built with Python, Machine Learning, FastAPI, and Streamlit.*
