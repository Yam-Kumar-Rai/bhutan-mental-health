
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Student Wellbeing | Bhutan",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "student_wellbeing_data.csv"
MODEL_PATH = BASE_DIR / "student_wellbeing_model.pkl"

# --------------------------------------------------
# 2. Custom CSS
# --------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F6F8F5;
    }

    [data-testid="stSidebar"] {
        background-color: #E8F0E8;
    }

    .main-title {
        font-size: 2.35rem;
        font-weight: 750;
        color: #244B3A;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #63756A;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: #FFFFFF;
        padding: 20px;
        border-radius: 14px;
        border: 1px solid #E0E8E0;
        box-shadow: 0 2px 8px rgba(30, 60, 40, 0.04);
    }

    .metric-label {
        color: #68796D;
        font-size: 0.9rem;
    }

    .metric-value {
        color: #244B3A;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 6px;
    }

    div.stButton > button[kind="primary"] {
        background-color: #327653;
        border: none;
        border-radius: 9px;
        min-height: 46px;
        font-weight: 600;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #24583D;
        color: white;
    }

    h1, h2, h3 {
        color: #244B3A;
    }

    .disclaimer {
        background: #FFF6DF;
        border-left: 4px solid #D9A441;
        padding: 13px 16px;
        border-radius: 7px;
        color: #67552D;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 3. Load model and dataset
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 4. Check files
# --------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH.name}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Model not found: {MODEL_PATH.name}")
    st.stop()

try:
    df = load_data()
    model = load_model()
except Exception as error:
    st.error("Could not load the dataset or saved model.")
    st.code(str(error))
    st.info(
        "Check that the saved model's scikit-learn version "
        "is compatible with the version in requirements.txt."
    )
    st.stop()


# --------------------------------------------------
# 5. Find the model's input columns
# --------------------------------------------------
model_features = getattr(model, "feature_names_in_", None)

if model_features is not None:
    feature_columns = list(model_features)
else:
    # Fallback: assumes the last CSV column is the target.
    # Update this if your model was trained differently.
    feature_columns = list(df.columns[:-1])

missing_features = [
    col for col in feature_columns if col not in df.columns
]

if missing_features:
    st.error(
        "The dataset is missing columns required by the model: "
        + ", ".join(missing_features)
    )
    st.stop()

# Guess a target column for displaying dataset insights.
target_candidates = [
    "Wellbeing_Level",
    "Wellbeing_Status",
    "Mental_Health_Status",
    "Mental_Health_Level",
    "Wellbeing",
    "Target",
    "target",
    "label"
]

target_column = next(
    (col for col in target_candidates if col in df.columns),
    None
)

if target_column is None:
    remaining_columns = [
        col for col in df.columns if col not in feature_columns
    ]
    if remaining_columns:
        target_column = remaining_columns[0]


# --------------------------------------------------
# 6. Sidebar navigation
# --------------------------------------------------
st.sidebar.markdown("## 🌿 Student Wellbeing")
st.sidebar.caption("Bhutan • Educational ML Project")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Student Assessment",
        "Data Insights",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Built with Python, scikit-learn and Streamlit")


# --------------------------------------------------
# 7. Reusable metric card
# --------------------------------------------------
def metric_card(label, value, description=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{description}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# 8. Overview dashboard
# --------------------------------------------------
if page == "Overview":

    st.markdown(
        '<div class="main-title">Student Wellbeing Dashboard 🌿</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">'
        'Exploring student wellbeing through a basic machine-learning '
        'demonstration using fictional data.'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="disclaimer"><b>Important:</b> This is a classroom '
        'demonstration using synthetic data. It is not a medical or '
        'mental health diagnostic tool.</div>',
        unsafe_allow_html=True
    )

    st.write("")

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Synthetic student records",
            f"{len(df):,}",
            "Rows in the sample dataset"
        )

    with c2:
        metric_card(
            "Input features",
            f"{len(feature_columns)}",
            "Variables used by the model"
        )

    with c3:
        metric_card(
            "Dataset columns",
            f"{len(df.columns)}",
            "Total variables available"
        )

    st.write("")
    st.subheader("Explore the dashboard")

    left, right = st.columns(2)

    with left:
        st.markdown("### 📝 Student assessment")
        st.write(
            "Enter a fictional student profile and view the saved "
            "model's predicted category."
        )
        if st.button("Open assessment →", use_container_width=True):
            st.session_state["page"] = "Student Assessment"
            st.rerun()

    with right:
        st.markdown("### 📊 Data insights")
        st.write(
            "Explore sample records, feature distributions and "
            "the available target categories."
        )
        if st.button("Explore data →", use_container_width=True):
            st.session_state["page"] = "Data Insights"
            st.rerun()

    if target_column is not None:
        st.write("")
        st.subheader("Dataset target distribution")

        counts = df[target_column].value_counts().head(12)

        fig, ax = plt.subplots(figsize=(9, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel(target_column)
        ax.set_ylabel("Number of records")
        ax.set_title("Distribution in the synthetic dataset")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 9. Student assessment page
# --------------------------------------------------
elif page == "Student Assessment":

    st.title("📝 Student Information")
    st.write(
        "Enter a fictional profile using values consistent with "
        "the dataset used to train the model."
    )

    st.info(
        "Use this form for educational experimentation only. "
        "Do not enter identifiable or private student information."
    )

    with st.form("student_assessment_form"):

        input_values = {}
        columns = st.columns(2)

        for index, feature in enumerate(feature_columns):
            series = df[feature].dropna()
            container = columns[index % 2]

            with container:
                if series.empty:
                    st.warning(f"No example values found for {feature}.")
                    input_values[feature] = None
                    continue

                if (
                    pd.api.types.is_numeric_dtype(series)
                    and not pd.api.types.is_bool_dtype(series)
                ):
                    minimum = float(series.min())
                    maximum = float(series.max())
                    median = float(series.median())

                    if minimum == maximum:
                        input_values[feature] = minimum
                        st.text_input(
                            feature,
                            value=str(minimum),
                            disabled=True
                        )
                    else:
                        # Use integer steps for integer-valued features.
                        is_integer = bool(
                            pd.api.types.is_integer_dtype(series)
                        )

                        input_values[feature] = st.slider(
                            feature,
                            min_value=(
                                int(minimum) if is_integer else minimum
                            ),
                            max_value=(
                                int(maximum) if is_integer else maximum
                            ),
                            value=(
                                int(median) if is_integer else median
                            ),
                            step=1 if is_integer else None,
                            help=f"Observed range in the dataset: "
                                 f"{minimum:g} to {maximum:g}"
                        )
                else:
                    choices = series.astype(str).unique().tolist()
                    input_values[feature] = st.selectbox(
                        feature,
                        options=choices
                    )

        submitted = st.form_submit_button(
            "Generate Model Prediction",
            type="primary",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = pd.DataFrame(
                [input_values],
                columns=feature_columns
            )

            # Restore numeric dtypes when possible.
            for feature in feature_columns:
                if pd.api.types.is_numeric_dtype(df[feature]):
                    input_df[feature] = pd.to_numeric(
                        input_df[feature]
                    )

            prediction = model.predict(input_df)[0]

            st.divider()
            st.subheader("Model output")

            st.success(f"Predicted category: **{prediction}**")

            st.caption(
                "This is the output of a machine-learning model trained "
                "on synthetic data. It does not establish a student's "
                "actual mental health or wellbeing."
            )

            if hasattr(model, "predict_proba"):
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    classes = model.classes_

                    probability_df = pd.DataFrame({
                        "Category": classes,
                        "Model probability (%)": probabilities * 100
                    }).sort_values(
                        "Model probability (%)",
                        ascending=False
                    )

                    st.markdown("#### Model probability estimates")
                    st.dataframe(
                        probability_df.style.format({
                            "Model probability (%)": "{:.1f}%"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                    st.caption(
                        "Probabilities reflect the model's estimates, "
                        "not clinically validated risk or certainty."
                    )
                except Exception:
                    st.caption(
                        "Probability estimates are unavailable for "
                        "this model configuration."
                    )

        except Exception as error:
            st.error("The model could not generate a prediction.")
            st.code(str(error))
            st.write(
                "The input fields must match the features and data "
                "types used when the model was trained."
            )


# --------------------------------------------------
# 10. Data insights page
# --------------------------------------------------
elif page == "Data Insights":

    st.title("📊 Data Insights")
    st.write(
        "Explore the fictional student records used in this "
        "educational demonstration."
    )

    st.markdown("### Sample dataset")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Showing up to 100 records. All records should be synthetic "
        "for this classroom project."
    )

    st.divider()
    st.markdown("### Feature distribution")

    available_features = [
        col for col in feature_columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    if available_features:
        selected_feature = st.selectbox(
            "Choose a numeric feature",
            available_features
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        df[selected_feature].dropna().plot(
            kind="hist",
            bins=20,
            ax=ax
        )
        ax.set_title(f"Distribution of {selected_feature}")
        ax.set_xlabel(selected_feature)
        ax.set_ylabel("Frequency")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No numeric features were detected for this chart.")

    if target_column is not None:
        st.divider()
        st.markdown("### Target category counts")

        counts = df[target_column].value_counts()

        fig, ax = plt.subplots(figsize=(9, 4))
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Distribution of {target_column}")
        ax.set_xlabel(target_column)
        ax.set_ylabel("Number of records")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 11. About project page
# --------------------------------------------------
elif page == "About Project":

    st.title("🌱 About This Project")

    st.markdown("""
    This application demonstrates how a simple machine-learning
    workflow can be connected to an interactive web interface.

    **Technologies used**
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Matplotlib

    **Educational workflow**
    1. Generate or load fictional student records.
    2. Prepare features for a machine-learning model.
    3. Train and save the model.
    4. Load the saved model in Streamlit.
    5. Collect example inputs and display model output.
    """)

    st.warning("""
    **Limitations and responsible use**

    The dataset is synthetic and may not represent real students in
    Bhutan. Model outputs are not clinical assessments, diagnoses,
    or validated mental health predictions. Do not use this app to
    make decisions about real students or to replace qualified
    professional support.
    """)

    st.caption("Student Wellbeing • Bhutan • Classroom ML Project")
