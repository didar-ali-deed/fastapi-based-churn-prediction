from pathlib import Path

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Number of customers
N = 4000

# Generate customer features
age = np.random.randint(18, 75, N)

tenure_months = np.random.randint(1, 73, N)

monthly_charges = np.round(np.random.uniform(25, 120, N), 2)

contract = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.25, 0.20]
)

internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.35, 0.45, 0.20]
)

support_calls = np.random.poisson(2, N)
support_calls = np.clip(support_calls, 0, 10)

paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.65, 0.35])

payment_method = np.random.choice(
    ["Electronic check", "Credit card", "Bank transfer", "Mailed check"], N
)

# Generate churn probability
churn_score = (
    -1.0
    + (contract == "Month-to-month") * 1.4
    - (contract == "Two year") * 1.0
    + (tenure_months < 12) * 0.9
    + (monthly_charges > 85) * 0.5
    + (support_calls >= 3) * 0.8
    + (internet_service == "Fiber optic") * 0.3
    + (paperless_billing == "Yes") * 0.2
    + (payment_method == "Electronic check") * 0.5
    + (age < 25) * 0.2
)

# Convert scores to probabilities
probability = 1 / (1 + np.exp(-churn_score))

# Generate binary target
churn = np.random.binomial(1, probability)

# Create dataset
df = pd.DataFrame(
    {
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "contract": contract,
        "internet_service": internet_service,
        "support_calls": support_calls,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "churn": churn,
    }
)

# Save dataset
output_path = DATA_DIR / "customer_churn.csv"
df.to_csv(output_path, index=False)

print("=" * 45)
print("DATASET GENERATED SUCCESSFULLY")
print("=" * 45)

print(f"Dataset shape: {df.shape}")
print(f"Total customers: {len(df)}")
print(f"Churned customers: {df['churn'].sum()}")
print(f"Non-churned customers: {(df['churn'] == 0).sum()}")
print(f"Churn percentage: {df['churn'].mean() * 100:.2f}%")

print("\nMissing values:")
print(df.isnull().sum())

print(f"\nSaved to: {output_path}")
