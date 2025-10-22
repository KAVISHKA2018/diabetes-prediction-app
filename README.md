# Diabetes Prediction Web Application

This project is a **machine learning-powered web application** that predicts whether a person is diabetic or non-diabetic based on medical diagnostic data.  
It is built using **Streamlit** for deployment and a trained **Random Forest Classifier** model.

---

## Features
- Interactive web interface built with **Streamlit**
- Real-time diabetes prediction using 8 health parameters
- Data exploration and filtering tools
- Visualizations of dataset trends and feature correlations
- Model performance metrics (accuracy, ROC, confusion matrix)
- User-friendly dashboard with sidebar navigation

---

## Project Structure

1. app.py              - Streamlit main application
2. model.pkl           - Trained Random Forest model
3. scaler.pkl          - StandardScaler for feature scaling
4. data/diabetes.csv   - Dataset (PIMA Indians Diabetes Database)
5. requirements.txt    - Python dependencies
6. README.md           - Project documentation

---

## Features Used

The Pima Indians Diabetes Dataset contains health metrics from 768 patients with 8 features:

| Feature                  | Description                  |
| ------------------------ | ---------------------------- |
| Pregnancies              | Number of times pregnant     |
| Glucose                  | Plasma glucose concentration |
| BloodPressure            | Diastolic blood pressure     |
| SkinThickness            | Triceps skin fold thickness  |
| Insulin                  | 2-hour serum insulin         |
| BMI                      | Body Mass Index              |
| DiabetesPedigreeFunction | Diabetes heredity score      |
| Age                      | Age in years                 |
| Outcome                  | 0 = No Diabetes, 1 = Diabetes|

---

## Installation

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---

## Model Training

The model was trained using a Random Forest classifier with 100 trees. The dataset was split into 80% training and 20% testing. Feature scaling was applied using StandardScaler.

- Algorithm: `Random Forest Classifier`
- Number of Trees: `100`
- Scaler Used: `StandardScaler`
- Dataset: `PIMA Indians Diabetes Database`
- Accuracy: `78.57% (Test Data)`
- Cross Validation: `77.86% (5-fold CV)`

---

## Deployment

The app can be deployed on Streamlit Cloud by connecting to this GitHub repository.

---

## Disclaimer

This web application is for educational and demonstration purposes only.
Predictions made by this model should not be used as medical advice.
Always consult a healthcare professional for accurate medical assessment.

---
