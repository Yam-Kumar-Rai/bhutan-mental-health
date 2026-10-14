# Mental Health and Student Wellbeing in Bhutan

## 1. Project Overview

This project demonstrates a basic machine-learning application for exploring student wellbeing using synthetic data representing 800 fictional student profiles.

The application uses a Logistic Regression model and a Streamlit interface to demonstrate how student-related factors can be used to generate model predictions.

## 2. Project Objectives

- Demonstrate a basic machine-learning workflow.
- Explore student-related factors using data analysis and visualization.
- Train and use a Logistic Regression classification model.
- Develop an interactive web application using Streamlit.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## 4. Project Files

- `app.py` — Main Streamlit application.
- `student_wellbeing_model.pkl` — Saved machine-learning model.
- `bhutan_student_wellbeing.csv` — Synthetic student dataset.
- `logistic_model.ipynb` — Notebook for model development and training.
- `requirements.txt` — Required Python libraries.
- `README.md` — Project documentation.

## 5. Installation

Install Python 3.11 and Git. Open a terminal in the project folder and run:

```bash
python -m pip install -r requirements.txt
```

## 6. Run the Application

Run the following command from the project folder:

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## 7. Dataset Description

The dataset contains 800 fictional student profiles. Features may include age, sleep hours, study hours, physical activity, stress level, social support, financial pressure, academic pressure, screen time, and mental health awareness.

The model generates predictions based on patterns in this synthetic dataset.

## 8. Limitations and Ethical Considerations

This project uses synthetic data for educational purposes. It does not represent real Bhutanese students, and its predictions are not validated assessments of a person's mental health.

The application is not a medical diagnostic tool and must not be used to make real mental health or student-support decisions.

## 9. Conclusion

This project demonstrates how data analysis, machine learning, and Streamlit can be combined to create an interactive academic application for exploring student wellbeing.