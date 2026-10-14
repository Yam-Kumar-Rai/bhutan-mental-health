
import streamlit as st
import pandas as pd
import joblib

# Configure the page
st.set_page_config(
    page_title="Student Wellbeing",
    page_icon="🌱",
    layout="centered"
)

# Load the saved machine-learning model
@st.cache_resource
def load_model():
    return joblib.load("student_wellbeing_model.pkl")

# Load the synthetic dataset
@st.cache_data
def load_data():
    return pd.read_csv("bhutan_student_wellbeing.csv")
try:
    model = load_model()
    df = load_data()
except FileNotFoundError:
    st.error(
        "Model or dataset file not found. "
        "Place both files in the same folder as app.py."
    )
    st.stop()

# Application title
st.title("Mental Health and Student Wellbeing in Bhutan")

st.write(
    "This educational application demonstrates a basic "
    "machine-learning workflow using fictional student data."
)

st.warning(
    "This is a classroom demonstration, not a medical "
    "or mental health diagnostic tool."
)

# Input form
st.header("Enter Student Information")

with st.form("student_form"):

    age = st.number_input(
        "Age", min_value=16, max_value=60, value=21
    )

    gender = st.selectbox(
        "Gender", ["Female", "Male", "Other"]
    )

    district = st.selectbox(
        "District", sorted(df["district"].dropna().unique())
    )

    residence = st.selectbox(
        "Residence", sorted(df["residence"].dropna().unique())
    )

    sleep_hours = st.slider(
        "Sleep hours per night", 0.0, 12.0, 7.0, 0.5
    )

    study_hours_per_day = st.slider(
        "Study hours per day", 0.0, 16.0, 4.0, 0.5
    )

    physical_activity_days = st.slider(
        "Physical activity days per week", 0, 7, 3
    )

    stress_level = st.slider(
        "Stress level (1–10)", 1, 10, 5
    )

    social_support = st.slider(
        "Social support (1–10)", 1, 10, 5
    )

    financial_pressure = st.slider(
        "Financial pressure (1–10)", 1, 10, 5
    )

    academic_pressure = st.slider(
        "Academic pressure (1–10)", 1, 10, 5
    )

    screen_time_hours = st.slider(
        "Screen time per day", 0.0, 16.0, 5.0, 0.5
    )

    mental_health_awareness = st.selectbox(
        "Mental health awareness",
        sorted(
            df["mental_health_awareness"]
            .dropna()
            .unique()
        )
    )

    submitted = st.form_submit_button("Run Prediction")

if submitted:
    # Create one row using the model's expected feature names
    student = pd.DataFrame([{
        "age": age,
        "gender": gender,
        "district": district,
        "residence": residence,
        "sleep_hours": sleep_hours,
        "study_hours_per_day": study_hours_per_day,
        "physical_activity_days": physical_activity_days,
        "stress_level": stress_level,
        "social_support": social_support,
        "financial_pressure": financial_pressure,
        "academic_pressure": academic_pressure,
        "screen_time_hours": screen_time_hours,
        "mental_health_awareness": mental_health_awareness
    }])

    prediction = model.predict(student)[0]

    st.subheader("Demonstration Result")
    st.write("Model output:", prediction)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(student)[0]
        classes = model.classes_

        results = pd.DataFrame({
            "Model output category": classes,
            "Predicted probability": probabilities
        })

        st.write("Model probabilities (not a clinical risk score):")
        st.dataframe(results, hide_index=True, use_container_width=True)

# Dataset preview
with st.expander("View sample synthetic data"):
    st.dataframe(df.head(10), use_container_width=True)
