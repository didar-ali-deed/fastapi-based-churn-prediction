from __future__ import annotations

import html
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Palette (all used with black text -> high contrast)
LOW, MEDIUM, HIGH = "#7CDB3A", "#F5C518", "#FF5C8A"
FONT = "Space Grotesk, Inter, sans-serif"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
st.html(
    """
<style>
/* Base */
.stApp { background: #FFFBEA; }
html, body, .stApp, p, label, span, li { font-family: Inter, sans-serif; font-size: 16px; color: #000; }
h2, h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 700 !important;
         color: #000 !important; text-transform: uppercase; }
h2 { font-size: 2.2rem !important; }
h3 { font-size: 1.4rem !important; }
hr { border-top: 3px solid #000 !important; opacity: 1; }
[data-testid="stCaptionContainer"] p { font-size: 0.9rem; color: #000; }

/* Boxes: thick border + flat shadow */
[class*="st-key-card"], [data-testid="stForm"], [data-testid="stMetric"],
[data-testid="stAlert"], [data-testid="stExpander"], [data-testid="stDataFrame"] {
    background: #fff; border: 3px solid #000 !important; border-radius: 0 !important;
    box-shadow: 5px 5px 0 #000;
}
[class*="st-key-card"] { padding: 20px; }
[data-testid="stMetric"] { padding: 12px 16px; }
[data-testid="stAlert"] * { color: #000; }

/* Sidebar */
[data-testid="stSidebar"] { background: #F5C518; border-right: 3px solid #000; }
[data-testid="stSidebar"] * { color: #000; }

/* Inputs */
[data-testid="stWidgetLabel"] p { font-weight: 700; font-size: 1rem; }
[data-baseweb="input"], [data-baseweb="select"] > div { border: 3px solid #000; border-radius: 0; }
[data-baseweb="input"], [data-baseweb="input"] > div, [data-baseweb="select"] > div, input {
    background: #fff !important; }
input, [data-baseweb="select"] * { color: #000 !important; font-size: 1rem; font-weight: 600; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] li { background: #fff !important; }
[data-baseweb="popover"] * { color: #000 !important; }
[data-testid="stNumberInput"] button { background: #F5C518 !important; color: #000 !important; }

/* Alerts, header, links */
[data-testid="stAlert"], [data-testid="stAlert"] > div { background: #fff !important; border-radius: 0 !important; }
[data-testid="stHeader"] { background: #FFFBEA; }
[data-testid="stHeader"] * { color: #000 !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { opacity: 1; color: #000 !important; }
a { color: #000 !important; font-weight: 700; }

/* Buttons */
.stButton button, .stFormSubmitButton button, .stDownloadButton button {
    background: #F5C518; border: 3px solid #000; border-radius: 0; box-shadow: 4px 4px 0 #000;
    font-family: 'Space Grotesk', sans-serif; font-weight: 700; text-transform: uppercase; }
.stButton button p, .stFormSubmitButton button p, .stDownloadButton button p {
    color: #000; font-size: 1rem; font-weight: 700; }
.stButton button:hover, .stFormSubmitButton button:hover, .stDownloadButton button:hover {
    background: #FF5C8A; border-color: #000; transform: translate(2px, 2px); box-shadow: 2px 2px 0 #000; }

/* Metrics */
[data-testid="stMetricLabel"] p { font-size: 0.9rem; font-weight: 600; text-transform: uppercase; }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 700; }

/* Custom elements */
.badge { display: inline-block; padding: 6px 14px; border: 3px solid #000; box-shadow: 3px 3px 0 #000;
         font: 700 1rem 'Space Grotesk', sans-serif; text-transform: uppercase; color: #000; }
.chip { display: inline-block; margin: 4px 6px 4px 0; padding: 4px 10px; border: 2px solid #000;
        background: #7CDB3A; font-weight: 600; font-size: 0.9rem; color: #000; }
</style>
"""
)

# ---------------- Helpers ----------------


@st.cache_data(ttl=10, show_spinner=False)
def check_api():
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_model_info():
    r = requests.get(f"{API_URL}/model/info", timeout=5)
    r.raise_for_status()
    return r.json()


def err_msg(e: requests.RequestException) -> str:
    r = getattr(e, "response", None)
    if r is not None:
        try:
            return str(r.json().get("detail", r.text))
        except ValueError:
            return r.text
    return str(e)


def risk_tier(p: float):
    if p < 0.35:
        return "Low risk", LOW
    if p < 0.65:
        return "Medium risk", MEDIUM
    return "High risk", HIGH


def style(fig: go.Figure, height: int, margin: dict) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=margin,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": "#000", "size": 14},
    )
    return fig


def gauge(p: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=p * 100,
            number={"suffix": "%", "font": {"size": 44}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#000", "tickwidth": 2},
                "bar": {"color": color, "thickness": 0.35},
                "bgcolor": "#fff",
                "borderwidth": 3,
                "bordercolor": "#000",
                "steps": [
                    {"range": [0, 35], "color": "#E8F8D4"},
                    {"range": [35, 65], "color": "#FFF3C4"},
                    {"range": [65, 100], "color": "#FFE0EA"},
                ],
            },
        )
    )
    return style(fig, 260, {"l": 20, "r": 20, "t": 20, "b": 0})


# ---------------- Sidebar ----------------

api_ok = bool((h := check_api()) and h.get("model_loaded"))

with st.sidebar:
    st.markdown("### Churn Predictor")
    page = st.radio(
        "Navigate", ["Prediction", "Dashboard"], label_visibility="collapsed"
    )
    st.divider()
    if api_ok:
        st.success("API connected")
    else:
        st.error("API disconnected")
    st.caption(f"Endpoint: {API_URL}")
    if st.button("Refresh status", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------------- Dashboard ----------------

if page == "Dashboard":
    st.header("Model overview")
    if not api_ok:
        st.error(
            "The FastAPI backend is unreachable. Start the server, then refresh the status."
        )
        st.stop()
    try:
        info = get_model_info()
    except requests.RequestException as e:
        st.error(f"Could not load model info: {err_msg(e)}")
        st.stop()

    for col, (label, key) in zip(
        st.columns(3),
        [("Model", "model_name"), ("Framework", "framework"), ("Task", "task")],
    ):
        col.metric(label, info[key])

    names = ["Accuracy", "Precision", "Recall", "F1 score"]
    values = [
        info["metrics"][k] for k in ("accuracy", "precision", "recall", "f1_score")
    ]
    for col, name, v in zip(st.columns(4), names, values):
        col.metric(name, f"{v * 100:.2f}%")

    left, right = st.columns([3, 2], gap="large")
    with left, st.container(key="card_perf"):
        st.subheader("Performance")
        fig = go.Figure(
            go.Bar(
                x=[v * 100 for v in values],
                y=names,
                orientation="h",
                marker_color=MEDIUM,
                marker_line=dict(color="#000", width=3),
                text=[f"{v * 100:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(size=15, color="#000"),
            )
        )
        fig.update_xaxes(range=[0, 115], visible=False)
        fig.update_yaxes(autorange="reversed", tickfont=dict(size=15, color="#000"))
        st.plotly_chart(
            style(fig, 300, dict(l=0, r=50, t=10, b=0)),
            use_container_width=True,
            theme=None,
        )
    with right, st.container(key="card_features"):
        st.subheader("Input features")
        features = info.get("features", [])
        if isinstance(features, list):
            st.markdown(
                "".join(
                    f'<span class="chip">{html.escape(str(f))}</span>' for f in features
                ),
                unsafe_allow_html=True,
            )
        else:
            st.json(features)

# ---------------- Prediction ----------------

else:
    st.header("Predict churn")
    st.write("Enter the customer profile and run the model.")
    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("history", [])

    form_col, result_col = st.columns([3, 2], gap="large")
    with form_col, st.form("prediction_form"):
        st.subheader("Customer profile")
        a, b = st.columns(2)
        with a:
            age = st.number_input("Age", 18, 120, 28)
            tenure = st.number_input("Tenure (months)", 1, 120, 12)
            monthly_charges = st.number_input(
                "Monthly charges", 0.0, value=75.5, step=0.5
            )
            support_calls = st.number_input("Support calls", 0, value=3)
        with b:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            billing = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Credit card", "Bank transfer", "Mailed check"],
            )
        submitted = st.form_submit_button("Predict churn", use_container_width=True)

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
            with st.spinner("Scoring customer..."):
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                r.raise_for_status()
                result = r.json()
            st.session_state.last_result = result
            st.session_state.history.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    **payload,
                    "label": result["label"],
                    "churn_probability": round(result["churn_probability"], 4),
                }
            )
        except requests.RequestException as e:
            st.session_state.last_result = None
            with result_col:
                st.error(f"Prediction failed: {err_msg(e)}")

    with result_col, st.container(key="card_result"):
        st.subheader("Result")
        if not (result := st.session_state.last_result):
            st.info("Submit the form to see the prediction.")
        else:
            p = result["churn_probability"]
            tier, color = risk_tier(p)
            st.markdown(
                f'<span class="badge" style="background:{color}">{tier}</span>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(gauge(p, color), use_container_width=True, theme=None)
            m1, m2 = st.columns(2)
            m1.metric("Prediction", result["label"])
            m2.metric("Churn probability", f"{p * 100:.2f}%")
            st.caption("Risk bands: under 35% low · 35–65% medium · over 65% high")
            with st.expander("API response"):
                st.json(result)

    if st.session_state.history:
        st.divider()
        head, action = st.columns([4, 1])
        head.subheader("Session history")
        df = pd.DataFrame(st.session_state.history)
        action.download_button(
            "Download CSV",
            df.to_csv(index=False).encode(),
            "churn_predictions.csv",
            "text/csv",
            use_container_width=True,
        )
        st.dataframe(df.iloc[::-1], use_container_width=True, hide_index=True)
