import requests
import streamlit as st

# ---------------------------------
# Configuration
# ---------------------------------

API_URL = "http://127.0.0.1:8000"


# ---------------------------------
# Page configuration
# ---------------------------------

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction")
st.caption("Machine Learning powered prediction dashboard")


# ---------------------------------
# API connection
# ---------------------------------


def check_api():

    try:
        response = requests.get(f"{API_URL}/health", timeout=5)

        response.raise_for_status()

        return response.json()

    except requests.RequestException:
        return None


health = check_api()

if health and health.get("model_loaded"):
    st.sidebar.success("API Connected")
else:
    st.sidebar.error("API Disconnected")


st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")

page = st.sidebar.radio("Select Page", ["Dashboard", "Prediction"])


# ---------------------------------
# Dashboard
# ---------------------------------

if page == "Dashboard":
    st.header("Model Overview")

    if not health or not health.get("model_loaded"):
        st.error("FastAPI backend is unavailable.")

    else:
        try:
            response = requests.get(f"{API_URL}/model/info", timeout=5)

            response.raise_for_status()
            info = response.json()

            st.subheader(info["model_name"])

            st.write(f"Framework: {info['framework']}")

            st.write(f"Task: {info['task']}")

            metrics = info["metrics"]

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")

            col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")

            col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")

            col4.metric("F1 Score", f"{metrics['f1_score'] * 100:.2f}%")

            st.subheader("Model Features")

            st.write(info["features"])

        except requests.RequestException as error:
            st.error(f"API request failed: {error}")


# ---------------------------------
# Prediction page
# ---------------------------------

elif page == "Prediction":
    st.header("Customer Churn Prediction")

    st.write("Enter customer information to predict churn.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=28)

            tenure = st.number_input(
                "Tenure (months)", min_value=1, max_value=120, value=12
            )

            monthly_charges = st.number_input(
                "Monthly Charges", min_value=0.0, value=75.5
            )

            support_calls = st.number_input("Support Calls", min_value=0, value=3)

        with col2:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )

            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

            billing = st.selectbox("Paperless Billing", ["Yes", "No"])

            payment = st.selectbox(
                "Payment Method",
                ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
            )

        submitted = st.form_submit_button("Predict Churn", use_container_width=True)

    if submitted:
        payload = {
            "age": age,
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "contract": contract,
            "internet_service": internet,
            "support_calls": support_calls,
            "paperless_billing": billing,
            "payment_method": payment,
        }

        try:
            with st.spinner("Generating prediction..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)

                response.raise_for_status()

                result = response.json()

            st.divider()

            st.subheader("Prediction Result")

            probability = result["churn_probability"]

            if result["prediction"] == 1:
                st.error("⚠️ Churn Predicted")

            else:
                st.success("✅ No Churn Predicted")

            col1, col2 = st.columns(2)

            col1.metric("Prediction", result["label"])

            col2.metric("Churn Probability", f"{probability * 100:.2f}%")

            st.progress(probability)

            st.caption("Probability is the model's estimated churn probability.")

            with st.expander("View API response"):
                st.json(result)

        except requests.RequestException as error:
            st.error(f"Prediction request failed: {error}")
