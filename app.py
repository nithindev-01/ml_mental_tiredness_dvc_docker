import os
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Mental Tiredness Prediction",
    page_icon="🧠",
    layout="centered"
)


# ---------------------------------------------------------------------------
# Custom styling (background, card, title, button)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Page background: deep indigo to purple to teal gradient */
    .stApp {
        background: linear-gradient(135deg, #1a1446 0%, #3b1f6e 45%, #0f6b7a 100%);
        background-attachment: fixed;
    }
    [data-testid="stHeader"] { background: transparent; }

    /* Frosted-glass card behind the whole form */
    .block-container {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 20px;
        padding: 2.5rem 2.5rem 3rem 2.5rem;
        margin-top: 1.5rem;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.35);
    }

    /* Text colours for readability on the dark gradient */
    h2, h3, p, label,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stMarkdownContainer"] p { color: #f1f1ff !important; }

    /* Gradient title */
    h1 {
        text-align: center;
        background: linear-gradient(90deg, #ffd86f, #fc6262, #c471f5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* Section headers get an accent underline */
    h3 {
        border-bottom: 2px solid rgba(255, 216, 111, 0.6);
        padding-bottom: 0.3rem;
        margin-top: 1.2rem;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff9a44, #fc6076);
        color: white;
        font-size: 1.1rem;
        font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 0;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(252, 96, 118, 0.45);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Load model (saved by the training script)
# Looks for model.pkl first (GitHub / Streamlit Cloud), then the joblib file.
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = ["model.pkl", "models/model.pkl", "best_mental_tiredness_model.joblib"]
MODEL_PATH = next(
    (os.path.join(BASE_DIR, f) for f in CANDIDATES if os.path.exists(os.path.join(BASE_DIR, f))),
    None,
)


@st.cache_resource
def load_model(path):
    # joblib.load also reads files saved with joblib.dump or pickle.dump
    return joblib.load(path)


if MODEL_PATH is None:
    st.error(
        "Model file not found. Upload `model.pkl` (or `best_mental_tiredness_model.joblib`) "
        "to the same folder as app.py."
    )
    st.stop()

model = load_model(MODEL_PATH)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🧠 Mental Tiredness Score Prediction")
st.write("Enter your daily habits and work details below:")

st.subheader("💼 Work Activity")
c1, c2 = st.columns(2)
with c1:
    number_of_decisions_made = st.number_input("Decisions Made", min_value=0, value=120, step=1)
    context_switch_count = st.number_input("Context Switches", min_value=0, value=8, step=1)
    notifications_received = st.number_input("Notifications Received", min_value=0, value=65, step=1)
    screen_time_min = st.number_input("Screen Time (min)", min_value=0.0, value=300.0, step=10.0)
with c2:
    deep_work_min = st.number_input("Deep Work (min)", min_value=0.0, value=95.0, step=5.0)
    task_complexity_avg = st.slider("Average Task Complexity", 1.0, 10.0, 5.5, 0.1)
    workload_score = st.slider("Workload Score", 1.0, 10.0, 5.7, 0.1)
    break_frequency = st.number_input("Breaks Taken", min_value=0, value=4, step=1)

st.subheader("😴 Health & Lifestyle")
c3, c4 = st.columns(2)
with c3:
    sleep_hours = st.slider("Sleep Hours", 0.0, 14.0, 6.9, 0.1)
    deep_sleep_pct = st.slider("Deep Sleep (%)", 0.0, 60.0, 19.0, 0.5)
    caffeine_mg = st.number_input("Caffeine (mg)", min_value=0.0, value=129.0, step=10.0)
with c4:
    hydration_l = st.number_input("Hydration (litres)", min_value=0.0, value=1.9, step=0.1)
    mood = st.selectbox("Mood", ["Happy", "Neutral", "Low"], index=1)

st.subheader("🏢 Environment")
c5, c6 = st.columns(2)
with c5:
    work_type = st.selectbox("Work Type", ["Office", "Remote", "Manual", "Student"])
    work_environment = st.selectbox("Work Environment", ["Quiet", "Moderate Noise", "Noisy"])
with c6:
    noise_level_db = st.number_input("Noise Level (dB)", min_value=0.0, value=48.0, step=1.0)
    temperature_c = st.number_input("Temperature (°C)", min_value=0.0, value=23.0, step=0.5)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict", type="primary"):

    input_data = pd.DataFrame({
        "number_of_decisions_made": [number_of_decisions_made],
        "context_switch_count": [context_switch_count],
        "notifications_received": [notifications_received],
        "screen_time_min": [screen_time_min],
        "deep_work_min": [deep_work_min],
        "task_complexity_avg": [task_complexity_avg],
        "caffeine_mg": [caffeine_mg],
        "break_frequency": [break_frequency],
        "sleep_hours": [sleep_hours],
        "deep_sleep_pct": [deep_sleep_pct],
        "hydration_l": [hydration_l],
        "mood": [mood],
        "work_type": [work_type],
        "work_environment": [work_environment],
        "noise_level_db": [noise_level_db],
        "temperature_c": [temperature_c],
        "workload_score": [workload_score],
    })

    # Same engineered features used during training
    input_data["deep_sleep_hours"] = input_data["sleep_hours"] * (input_data["deep_sleep_pct"] / 100.0)
    input_data["workload_intensity"] = input_data["workload_score"] * input_data["context_switch_count"]

    prediction = float(model.predict(input_data)[0])
    prediction = max(prediction, 0.0)  # score cannot be negative

    st.success(f"Predicted Mental Tiredness Score: {prediction:.2f}")

    # Indicative level (dataset scores range roughly from 0 to 77)
    if prediction < 20:
        st.info("🟢 Low tiredness: you seem well rested.")
    elif prediction < 40:
        st.warning("🟡 Moderate tiredness: consider taking regular breaks.")
    else:
        st.error("🔴 High tiredness: rest, sleep and fewer context switches may help.")