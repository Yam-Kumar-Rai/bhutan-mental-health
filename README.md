# Mental Health and Student Wellbeing in Bhutan

## 1. Project Overview

This project demonstrates a basic machine-learning application for exploring student wellbeing using synthetic data representing 800 fictional student profiles.

The application uses Logistic Regression to demonstrate how machine learning can generate predictions based on student-related factors. A Streamlit interface allows users to enter information and view the model output.

## 2. Project Objectives

- Demonstrate a complete basic machine-learning workflow.
- Explore student-related factors using data analysis and visualization.
- Train a Logistic Regression classification model.
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
- `student_wellbeing_model.pkl` — Saved machine-learning model pipeline.
- `student_wellbeing_data.csv` — Synthetic dataset.
- `requirements.txt` — Required Python libraries.
- `README.md` — Project documentation.

## 5. Installation

Install Python, then open a terminal in the project folder and run:

```bash
pip install -r requirements.txt
```

## 6. Run the Application

Run the following command in the project folder:

```bash
streamlit run app.py
```

The application should open in your web browser.

## 7. Dataset Description

The dataset contains 800 fictional student profiles and includes features such as age, sleep hours, study hours, physical activity, stress level, social support, financial pressure, academic pressure, screen time, and mental health awareness.

The target variable represents the model's generated category: Support recommended or No immediate flag.

## 8. Limitations and Ethical Considerations

This project uses synthetic data for educational purposes. It does not represent real Bhutanese students, and its predictions should not be interpreted as reliable assessments of a person's mental health.

The application is not a medical diagnostic tool and should not be used to make real decisions about students' mental health.

## 9. Conclusion

This project demonstrates how data analysis, machine learning, and Streamlit can be combined to create a simple interactive application for an academic project.