import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)


# 1. Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)


# 2. Separate features and target
X = df.drop(columns=["churn"])
y = df["churn"]


# 3. Define feature types
numerical_features = ["age", "tenure_months", "monthly_charges", "support_calls"]

categorical_features = [
    "contract",
    "internet_service",
    "paperless_billing",
    "payment_method",
]


# 4. Build preprocessing pipeline
numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# 5. Define classifier
classifier = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


# 6. Combine preprocessing and model
model_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", classifier)]
)


# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


# 8. Train model
print("\nTraining Random Forest model...")

model_pipeline.fit(X_train, y_train)


# 9. Make predictions
y_pred = model_pipeline.predict(X_test)

y_probability = model_pipeline.predict_proba(X_test)[:, 1]


# 10. Evaluate model
metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, y_probability)),
}


# 11. Display results
print("\n" + "=" * 45)
print("MODEL EVALUATION")
print("=" * 45)

for name, value in metrics.items():
    print(f"{name.upper():15}: {value:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# 12. Save trained pipeline
model_path = MODEL_DIR / "churn_pipeline.joblib"

joblib.dump(model_pipeline, model_path)


# 13. Save evaluation metrics
metrics_path = MODEL_DIR / "metrics.json"

with open(metrics_path, "w") as file:
    json.dump(metrics, file, indent=4)


print("\n" + "=" * 45)
print("MODEL TRAINING COMPLETED")
print("=" * 45)

print(f"Model saved: {model_path}")
print(f"Metrics saved: {metrics_path}")
