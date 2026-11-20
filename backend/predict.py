import pandas as pd


def predict_customer(model, customer):
    """
    Generate a churn prediction for one customer.
    """
    # Convert Pydantic data into a dictionary
    customer_data = customer.model_dump()

    # Convert dictionary into DataFrame
    input_df = pd.DataFrame([customer_data])

    # Model prediction
    prediction = int(model.predict(input_df)[0])

    # Probability of churn (class 1)
    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)
    churn_index = classes.index(1)

    churn_probability = float(probabilities[churn_index])

    # Human-readable result
    label = "Churn" if prediction == 1 else "No Churn"

    return {
        "prediction": prediction,
        "label": label,
        "churn_probability": round(churn_probability, 4),
        "status": "success",
    }
