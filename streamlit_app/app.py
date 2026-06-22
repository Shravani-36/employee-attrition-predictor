import streamlit as st
import pandas as pd
import pickle

# Page Configuration
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom Styling
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

.metric-container {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Load Model
with open("../models/xgb_model_streamlit.pkl", "rb") as file:
    model = pickle.load(file)

# Load Scaler
with open("../models/scaler_streamlit.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load Feature Names
with open("../models/feature_names_streamlit.pkl", "rb") as file:
    feature_names = pickle.load(file)

# Sidebar
st.sidebar.title("📊 Employee Attrition Predictor")

st.sidebar.info(
    """
    Predict employee attrition risk using
    Machine Learning and XGBoost.
    """
)

st.sidebar.success("Model: XGBoost")

# Main Title
st.title("📊 Employee Attrition Predictor")

st.write(
    "Predict whether an employee is likely to leave the company based on employee-related factors."
)

st.divider()

# Input Layout
col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=1000,
        max_value=150000,
        value=10000
    )

    distance = st.number_input(
        "Distance From Home",
        min_value=1,
        max_value=30,
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
        max_value=4,
        value=3
    )

    work_life = st.slider(
        "Work Life Balance ⭐",
        min_value=1,
        max_value=4,
        value=3
    )

    env_sat = st.slider(
        "Environment Satisfaction ⭐",
        min_value=1,
        max_value=4,
        value=3
    )

    overtime = st.selectbox(
        "OverTime",
        ["No", "Yes"]
    )

st.divider()

# Prediction Button
if st.button("🚀 Predict Attrition"):

    overtime_value = 1 if overtime == "Yes" else 0

    # Create Input Data
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

    # Scale Input
    input_scaled = scaler.transform(input_data)

    # Predict Probability
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("📈 Attrition Risk Score")

    st.progress(float(probability))

    st.metric(
        label="Risk Probability",
        value=f"{probability:.2%}"
    )

    # Risk Category
    if probability < 0.30:

        st.success("🟢 Low Attrition Risk")

    elif probability < 0.60:

        st.warning("🟡 Medium Attrition Risk")

    else:

        st.error("🔴 High Attrition Risk")

    st.divider()

    # Recommendations
    st.subheader("💡 Recommendations")

    if probability > 0.60:

        st.write("• Reduce overtime workload")

        st.write("• Improve work-life balance")

        st.write("• Increase employee engagement")

        st.write("• Conduct employee satisfaction reviews")

        st.write("• Review compensation and growth opportunities")

    elif probability > 0.30:

        st.write("• Monitor employee satisfaction")

        st.write("• Provide career growth opportunities")

        st.write("• Encourage manager feedback sessions")

        st.write("• Improve employee recognition programs")

    else:

        st.write("• Employee appears stable")

        st.write("• Continue current retention practices")

        st.write("• Maintain positive work environment")

        st.write("• Encourage continuous professional development")

    st.divider()

    st.subheader("📋 Employee Summary")

    st.write(f"**Age:** {age}")
    st.write(f"**Monthly Income:** ₹{monthly_income:,}")
    st.write(f"**Distance From Home:** {distance} km")
    st.write(f"**Years At Company:** {years_company}")
    st.write(f"**Total Working Years:** {total_years}")
    st.write(f"**OverTime:** {overtime}")