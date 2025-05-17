import streamlit as st
import requests
import json

st.title("Diabetes Prediction")

st.write("Enter patient data to predict diabetes risk:")

# Define input fields based on dataset features
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose", min_value=0, max_value=200, value=120)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin", min_value=0, max_value=900, value=79)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
)
age = st.number_input("Age", min_value=0, max_value=120, value=33)

if st.button("Predict"):
    # Prepare the data as a list of lists (batch size 1)
    data = {
        "inputs": [
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age,
            ]
        ]
    }

    # Replace with your MLflow model server URL
    url = "http://localhost:1234/invocations"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, data=json.dumps(data), headers=headers)
        response.raise_for_status()
        prediction = response.json()
        pred_label = "Diabetic" if prediction["predictions"][0] == 1 else "Non-Diabetic"
        st.success(f"Prediction: **{pred_label}**")
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
