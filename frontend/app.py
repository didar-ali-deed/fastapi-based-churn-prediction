"""Customer Churn Prediction dashboard (Streamlit client for the FastAPI backend).
Neo-Brutalism design - refined contrast & typography.

Fixes applied:
- Fonts now load via <link> tags (reliable in Streamlit) instead of @import.
- All native Streamlit widgets (inputs, selectboxes, dropdown menus, bordered
  containers, dataframe, expander, divider) are overridden to match the palette.
"""

from __future__ import annotations

import html
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ---------------------------------
# Configuration
# ---------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# Refined Neo-Brutalism palette (strong contrast, controlled saturation)
INK = "#111111"  # near-black for text & borders
PAPER = "#F8F5EE"  # warm paper background
SURFACE = "#FFFFFF"  # pure white cards
YELLOW = "#F5C518"  # primary accent (less neon)
LIME = "#7CDB3A"  # success / low risk
PINK = "#FF5C8A"  # high risk / alert
SKY = "#3B82F6"  # info
MUTED = "#4B5563"  # secondary text (good contrast on paper)

# Risk colors
LOW = "#7CDB3A"
MEDIUM = "#F5C518"
HIGH = "#FF5C8A"

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------
# Fonts - load BEFORE any custom CSS
# (@import inside <style> is unreliable in Streamlit; <link> is not)
# ---------------------------------

st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# Styling - Neo-Brutalism (refined)
# ---------------------------------

st.markdown(
    f"""
    <style>
    :root {{
        --nb-ink: {INK};
        --nb-paper: {PAPER};
        --nb-surface: {SURFACE};
        --nb-yellow: {YELLOW};
        --nb-lime: {LIME};
        --nb-pink: {PINK};
        --nb-muted: {MUTED};
    }}

    html, body, .stApp, [class*="css"] {{
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        background-color: {PAPER} !important;
        color: {INK};
    }}

    /* Base text colors (no !important so component overrides still win) */
    .stApp p, .stApp span, .stApp label, .stApp li, .stApp small, .stApp td, .stApp th {{
        color: {INK};
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}
    .block-container {{
        padding-top: 1.4rem;
        max-width: 1160px;
    }}

    /* Hero */
    .hero {{
        background: {YELLOW};
        border: 3px solid {INK};
        box-shadow: 6px 6px 0 {INK};
        border-radius: 2px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.6rem;
        color: {INK};
    }}
    .hero h1 {{
        margin: 0;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {INK};
    }}
    .hero p {{
        margin: 0.35rem 0 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        font-weight: 500;
        color: {MUTED};
    }}

    /* Metrics */
    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border: 2.5px solid {INK};
        box-shadow: 4px 4px 0 {INK};
        border-radius: 2px;
        padding: 0.95rem 1.1rem;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {MUTED};
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
    }}
    div[data-testid="stMetricValue"] {{
        color: {INK};
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }}
    div[data-testid="stMetricDelta"] {{
        color: {MUTED};
    }}

    /* Forms */
    div[data-testid="stForm"] {{
        background: {SURFACE};
        border: 2.5px solid {INK};
        box-shadow: 5px 5px 0 {INK};
        border-radius: 2px;
        padding: 1.35rem;
    }}

    /* Widget labels above inputs */
    div[data-testid="stWidgetLabel"] p {{
        color: {INK} !important;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.85rem;
    }}

    /* FIX - number / text inputs: kill default gray styling */
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {{
        background-color: {SURFACE} !important;
        color: {INK} !important;
        -webkit-text-fill-color: {INK} !important;
        border: 2px solid {INK} !important;
        border-radius: 2px !important;
        box-shadow: 3px 3px 0 {INK};
        font-family: 'Inter', sans-serif !important;
        font-weight: 500;
    }}
    div[data-testid="stNumberInput"] input:focus,
    div[data-testid="stTextInput"] input:focus {{
        box-shadow: 3px 3px 0 {INK};
    }}
    /* number-input stepper buttons */
    div[data-testid="stNumberInput"] button {{
        background-color: {SURFACE} !important;
        border-color: {INK} !important;
        color: {INK} !important;
    }}
    div[data-testid="stNumberInput"] button:hover {{
        background-color: {YELLOW} !important;
        color: {INK} !important;
    }}

    /* FIX - selectboxes: white surface, ink border, ink text */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {{
        background-color: {SURFACE} !important;
        border: 2px solid {INK} !important;
        border-radius: 2px !important;
        box-shadow: 3px 3px 0 {INK};
    }}
    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {{
        color: {INK} !important;
        -webkit-text-fill-color: {INK} !important;
        font-family: 'Inter', sans-serif;
    }}
    /* FIX - dropdown menu (portaled to body) */
    div[data-baseweb="popover"] ul {{
        background-color: {SURFACE} !important;
        border: 2px solid {INK} !important;
        border-radius: 2px !important;
        box-shadow: 4px 4px 0 {INK};
    }}
    div[data-baseweb="popover"] li {{
        color: {INK} !important;
        font-family: 'Inter', sans-serif;
    }}
    div[data-baseweb="popover"] li:hover {{
        background-color: {YELLOW} !important;
        color: {INK} !important;
    }}
    div[data-baseweb="popover"] li[aria-selected="true"] {{
        background-color: {YELLOW} !important;
        color: {INK} !important;
    }}

    /* FIX - bordered containers (st.container(border=True)) */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: {SURFACE} !important;
        border: 2.5px solid {INK} !important;
        box-shadow: 4px 4px 0 {INK};
        border-radius: 2px;
    }}

    /* Primary buttons */
    .stButton > button,
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stDownloadButton"] > button {{
        background: {YELLOW} !important;
        color: {INK} !important;
        border: 2.5px solid {INK} !important;
        border-radius: 2px !important;
        box-shadow: 4px 4px 0 {INK} !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1.15rem !important;
        transition: transform 70ms, box-shadow 70ms !important;
    }}
    .stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        transform: translate(2px, 2px);
        box-shadow: 2px 2px 0 {INK} !important;
        background: {YELLOW} !important;
        color: {INK} !important;
    }}
    .stButton > button:active,
    div[data-testid="stFormSubmitButton"] > button:active {{
        transform: translate(4px, 4px);
        box-shadow: 0 0 0 {INK} !important;
    }}

    /* Chips */
    .chip {{
        display: inline-block;
        background: {SURFACE};
        color: {INK};
        border: 2px solid {INK};
        box-shadow: 3px 3px 0 {INK};
        border-radius: 2px;
        padding: 0.28rem 0.75rem;
        margin: 0.22rem 0.3rem 0.22rem 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
    }}

    /* Badges */
    .badge {{
        display: inline-block;
        border: 2.5px solid {INK};
        box-shadow: 3px 3px 0 {INK};
        border-radius: 2px;
        padding: 0.32rem 0.95rem;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 0.9rem;
        color: {INK};
    }}

    .muted {{
        color: {MUTED};
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
    }}

    /* FIX - divider */
    hr {{
        border: none !important;
        border-top: 2px solid {INK} !important;
        margin: 1.2rem 0;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {PAPER} !important;
        border-right: 3px solid {INK};
    }}
    section[data-testid="stSidebar"] .stRadio label p {{
        color: {INK} !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] [data-baseweb="radio"] div {{
        border-color: {INK} !important;
    }}
    section[data-testid="stSidebar"] [data-baseweb="radio"] svg {{
        fill: {INK} !important;
    }}

    /* Headings */
    h1, h2, h3 {{
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        color: {INK} !important;
    }}

    /* Expander */
    div[data-testid="stExpander"] {{
        border: 2.5px solid {INK};
        box-shadow: 4px 4px 0 {INK};
        border-radius: 2px;
        background: {SURFACE};
    }}
    div[data-testid="stExpander"] summary p {{
        color: {INK} !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
    }}
    div[data-testid="stExpander"] summary:hover {{
        background: {YELLOW};
    }}

    /* FIX - Dataframe wrapper */
    div[data-testid="stDataFrame"] {{
        border: 2.5px solid {INK};
        box-shadow: 4px 4px 0 {INK};
        border-radius: 2px;
        background: {SURFACE};
    }}

    /* Alerts / captions / spinner */
    div[data-testid="stAlert"], div[data-testid="stNotification"] {{
        border: 2px solid {INK} !important;
        box-shadow: 3px 3px 0 {INK};
        border-radius: 2px;
    }}
    div[data-testid="stCaptionContainer"] {{
        color: {MUTED};
    }}
    div[data-testid="stSpinner"] {{
        color: {INK} !important;
    }}
    div[data-testid="stJson"] {{
        color: {INK};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Customer Churn Prediction</h1>
        <p>Random Forest classifier · FastAPI backend · Score a customer in seconds</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------
# API helpers
# ---------------------------------


@st.cache_data(ttl=10, show_spinner=False)
def check_api() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def get_model_info() -> dict:
    response = requests.get(f"{API_URL}/model/info", timeout=5)
    response.raise_for_status()
    return response.json()


def request_error_message(error: requests.RequestException) -> str:
    response = getattr(error, "response", None)
    if response is not None:
        try:
            return str(response.json().get("detail", response.text))
        except ValueError:
            return response.text
    return str(error)


def risk_tier(probability: float) -> tuple[str, str]:
    if probability < 0.35:
        return "Low risk", LOW
    if probability < 0.65:
        return "Medium risk", MEDIUM
    return "High risk", HIGH


def gauge(probability: float, color: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={
                "suffix": "%",
                "font": {
                    "size": 40,
                    "color": INK,
                    "family": "Space Grotesk, Inter, sans-serif",
                },
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": INK,
                    "tickwidth": 2,
                    "tickfont": {"color": MUTED, "size": 11},
                },
                "bar": {"color": color, "thickness": 0.30},
                "bgcolor": "#F1F1F1",
                "borderwidth": 2.5,
                "bordercolor": INK,
                "steps": [
                    {"range": [0, 35], "color": "#E8F8D4"},  # soft lime
                    {"range": [35, 65], "color": "#FFF3C4"},  # soft yellow
                    {"range": [65, 100], "color": "#FFE0EA"},  # soft pink
                ],
            },
        )
    )
    fig.update_layout(
        height=255,
        margin=dict(l=18, r=18, t=18, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, sans-serif", "color": INK},
    )
    return fig


# ---------------------------------
# Sidebar
# ---------------------------------

health = check_api()
api_ok = bool(health and health.get("model_loaded"))

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

# ---------------------------------
# Dashboard
# ---------------------------------

if page == "Dashboard":
    st.header("Model overview")

    if not api_ok:
        st.error(
            "The FastAPI backend is unreachable. Start the server, then refresh the status."
        )
        st.stop()

    try:
        info = get_model_info()
    except requests.RequestException as error:
        st.error(f"Could not load model info: {request_error_message(error)}")
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Model", info["model_name"])
    c2.metric("Framework", info["framework"])
    c3.metric("Task", info["task"])

    metrics = info["metrics"]
    names = ["Accuracy", "Precision", "Recall", "F1 score"]
    values = [metrics[k] for k in ("accuracy", "precision", "recall", "f1_score")]

    cols = st.columns(4)
    for col, name, value in zip(cols, names, values):
        col.metric(name, f"{value * 100:.2f}%")

    left, right = st.columns([3, 2], gap="large")

    with left, st.container(border=True):
        st.subheader("Performance")
        fig = go.Figure(
            go.Bar(
                x=[v * 100 for v in values],
                y=names,
                orientation="h",
                marker_color=YELLOW,
                marker_line=dict(color=INK, width=2),
                text=[f"{v * 100:.1f}%" for v in values],
                textposition="outside",
                textfont=dict(family="Inter, sans-serif", color=INK, size=13),
            )
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=40, t=10, b=0),
            xaxis=dict(range=[0, 112], showgrid=False, visible=False),
            yaxis=dict(
                autorange="reversed",
                tickfont=dict(
                    family="Space Grotesk, Inter, sans-serif", color=INK, size=13
                ),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif", "color": INK},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right, st.container(border=True):
        st.subheader("Input features")
        features = info.get("features", [])
        if isinstance(features, list):
            chips = "".join(
                f'<span class="chip">{html.escape(str(f))}</span>' for f in features
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.json(features)

# ---------------------------------
# Prediction
# ---------------------------------

else:
    st.header("Predict churn")
    st.markdown(
        '<p class="muted">Enter the customer profile and run the model.</p>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("last_result", None)
    st.session_state.setdefault("history", [])

    form_col, result_col = st.columns([3, 2], gap="large")

    with form_col, st.form("prediction_form"):
        st.subheader("Customer profile")
        a, b = st.columns(2)
        with a:
            age = st.number_input("Age", min_value=18, max_value=120, value=28)
            tenure = st.number_input(
                "Tenure (months)", min_value=1, max_value=120, value=12
            )
            monthly_charges = st.number_input(
                "Monthly charges", min_value=0.0, value=75.5, step=0.5
            )
            support_calls = st.number_input("Support calls", min_value=0, value=3)
        with b:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
            billing = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Credit card",
                    "Bank transfer",
                    "Mailed check",
                ],
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
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                response.raise_for_status()
                result = response.json()
            st.session_state.last_result = {"payload": payload, "result": result}
            st.session_state.history.append(
                {
                    "time": datetime.now().strftime("%H:%M:%S"),
                    **payload,
                    "label": result["label"],
                    "churn_probability": round(result["churn_probability"], 4),
                }
            )
        except requests.RequestException as error:
            st.session_state.last_result = None
            with result_col:
                st.error(f"Prediction failed: {request_error_message(error)}")

    with result_col:
        last = st.session_state.last_result
        with st.container(border=True):
            st.subheader("Result")
            if not last:
                st.info("Submit the form to see the prediction.")
            else:
                result = last["result"]
                probability = result["churn_probability"]
                tier, color = risk_tier(probability)

                st.markdown(
                    f'<span class="badge" style="background:{color}">{tier}</span>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(gauge(probability, color), use_container_width=True)

                m1, m2 = st.columns(2)
                m1.metric("Prediction", result["label"])
                m2.metric("Churn probability", f"{probability * 100:.2f}%")

                st.caption("Risk bands: under 35% low · 35–65% medium · over 65% high")
                with st.expander("API response"):
                    st.json(result)

    if st.session_state.history:
        st.divider()
        head, action = st.columns([4, 1])
        head.subheader("Session history")
        history_df = pd.DataFrame(st.session_state.history)
        action.download_button(
            "Download CSV",
            history_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(history_df.iloc[::-1], use_container_width=True, hide_index=True)
