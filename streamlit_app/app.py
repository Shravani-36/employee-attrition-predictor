import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# Custom Styling
# ----------------------------

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}

.stButton > button {
    width: 100%;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

div[data-testid="metric-container"] {
    border: 1px solid #e6e6e6;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Load Files
# ----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "xgb_model_streamlit.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler_streamlit.pkl"
FEATURE_PATH = BASE_DIR / "models" / "feature_names_streamlit.pkl"

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

with open(FEATURE_PATH, "rb") as file:
    feature_names = pickle.load(file)

# ----------------------------
# Sidebar
# ----------------------------

st.sidebar.title("📊 Employee Attrition Predictor")

st.sidebar.info(
    """
    Predict employee attrition risk using
    Machine Learning and XGBoost.
    """
)

st.sidebar.success("Model: XGBoost")

# ----------------------------
# Main Title
# ----------------------------

st.title("📊 Employee Attrition Predictor")

st.write(
    """
    Predict whether an employee is likely to leave the company
    based on employee-related factors.
    """
)

st.divider()

# ----------------------------
# Inputs
# ----------------------------

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    monthly_income = st.number_input(
        "Monthly Income (₹)",
        min_value=1000,
        max_value=150000,
        value=10000
    )

    distance = st.number_input(
        "Distance From Home (km)",
        min_value=1,
        max_value=50,
        value=5
    )

    years_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=5
    )

    total_years = st.number_input(
        "Total Working Years",
        min_value=0,
        max_value=40,
        value=8
    )

with col2:

    job_satisfaction = st.slider(
        "Job Satisfaction ⭐",
        min_value=1,
        max_value=5,
        value=3
    )

    work_life = st.slider(
        "Work Life Balance ⭐",
        min_value=1,
        max_value=5,
        value=3
    )

    env_sat = st.slider(
        "Environment Satisfaction ⭐",
        min_value=1,
        max_value=5,
        value=3
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

st.divider()

# ----------------------------
# Prediction
# ----------------------------

if st.button("🚀 Predict Attrition"):

    overtime_value = 1 if overtime == "Yes" else 0

    input_data = pd.DataFrame(
        [[
            age,
            monthly_income,
            distance,
            job_satisfaction,
            work_life,
            env_sat,
            years_company,
            total_years,
            overtime_value
        ]],
        columns=feature_names
    )

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📈 Attrition Risk Score")

    st.progress(float(probability))

    st.metric(
        label="Risk Probability",
        value=f"{probability:.2%}"
    )

    # Risk Level

    if probability < 0.30:

        st.success("🟢 Low Attrition Risk")

    elif probability < 0.60:

        st.warning("🟡 Medium Attrition Risk")

    else:

        st.error("🔴 High Attrition Risk")

    st.divider()

    # ----------------------------
    # Recommendations
    # ----------------------------

    st.subheader("💡 Recommendations")

    if probability > 0.60:

        st.write("• Reduce overtime workload")
        st.write("• Improve work-life balance")
        st.write("• Increase employee engagement")
        st.write("• Conduct employee satisfaction reviews")
        st.write("• Review compensation and career growth opportunities")

    elif probability > 0.30:

        st.write("• Monitor employee satisfaction")
        st.write("• Provide career growth opportunities")
        st.write("• Encourage manager feedback sessions")
        st.write("• Improve employee recognition programs")

    else:

        st.write("• Employee appears stable")
        st.write("• Continue current retention practices")
        st.write("• Maintain a positive work environment")
        st.write("• Encourage professional development")

    st.divider()

    # ----------------------------
    # Employee Summary
    # ----------------------------

    st.subheader("📋 Employee Summary")

    st.write(f"**Age:** {age}")
    st.write(f"**Monthly Income:** ₹{monthly_income:,}")
    st.write(f"**Distance From Home:** {distance} km")
    st.write(f"**Job Satisfaction:** {job_satisfaction}/5")
    st.write(f"**Work Life Balance:** {work_life}/5")
    st.write(f"**Environment Satisfaction:** {env_sat}/5")
    st.write(f"**Years At Company:** {years_company}")
    st.write(f"**Total Working Years:** {total_years}")
    st.write(f"**OverTime:** {overtime}")