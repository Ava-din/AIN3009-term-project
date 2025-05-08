"""
Streamlit app for diabetes prediction.
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import MODEL_NAME
from src.utils.mlflow_utils import setup_mlflow

# Set page title and icon
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🏥", layout="wide")

# App title
st.title("Diabetes Prediction App")
st.markdown("### Predict diabetes risk using the UCI Diabetes Dataset")
st.markdown("---")


# Function to load the model
@st.cache_resource
def load_production_model():
    """Load the latest production model from MLflow."""
    try:
        # Set up MLflow
        setup_mlflow()

        # Get the client
        client = mlflow.tracking.MlflowClient()

        # Get production models
        production_models = client.get_latest_versions(
            MODEL_NAME, stages=["Production"]
        )

        if not production_models:
            st.error(f"No production model found for {MODEL_NAME}")
            return None

        # Load the model
        model_uri = f"models:/{MODEL_NAME}/{production_models[0].version}"
        model = mlflow.sklearn.load_model(model_uri)

        model_info = {
            "name": production_models[0].name,
            "version": production_models[0].version,
            "run_id": production_models[0].run_id,
        }

        return model, model_info

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


# Load model information
model, model_info = load_production_model()

# Show model information
if model_info:
    st.sidebar.header("Model Information")
    st.sidebar.write(f"Model Name: {model_info['name']}")
    st.sidebar.write(f"Version: {model_info['version']}")
    st.sidebar.write(f"Run ID: {model_info['run_id']}")

# Create two columns
col1, col2 = st.columns([1, 1])

# Input form in first column
with col1:
    st.header("Patient Information")

    # Create form for patient information
    with st.form("patient_form"):
        # Basic information
        pregnancies = st.slider("Number of Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose Level (mg/dL)", 0, 300, 120)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 200, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
        insulin = st.slider("Insulin Level (mu U/ml)", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 60.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 20, 100, 35)

        # Calculate derived features
        glucose_insulin_ratio = glucose / (insulin + 1)

        # BMI Categories
        bmi_cat_underweight = 1 if bmi < 18.5 else 0
        bmi_cat_normal = 1 if 18.5 <= bmi < 25 else 0
        bmi_cat_overweight = 1 if 25 <= bmi < 30 else 0
        bmi_cat_obese = 1 if bmi >= 30 else 0

        # Submit button
        submit_button = st.form_submit_button("Predict Diabetes Risk")

# Results in second column
with col2:
    st.header("Prediction Results")

    if submit_button:
        if model is None:
            st.error("Model not loaded. Please check MLflow setup.")
        else:
            # Create input data
            input_data = {
                "Pregnancies": pregnancies,
                "Glucose": glucose,
                "BloodPressure": blood_pressure,
                "SkinThickness": skin_thickness,
                "Insulin": insulin,
                "BMI": bmi,
                "DiabetesPedigreeFunction": dpf,
                "Age": age,
                "Glucose_Insulin_Ratio": glucose_insulin_ratio,
                "BMI_Cat_Underweight": bmi_cat_underweight,
                "BMI_Cat_Normal": bmi_cat_normal,
                "BMI_Cat_Overweight": bmi_cat_overweight,
                "BMI_Cat_Obese": bmi_cat_obese,
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])

            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][prediction]

            # Display result
            if prediction == 1:
                st.error(f"Result: **POSITIVE** for diabetes")
                st.error(f"Probability: {probability:.2%}")
                st.warning(
                    "The model predicts that this patient has a high risk of diabetes."
                )
            else:
                st.success(f"Result: **NEGATIVE** for diabetes")
                st.success(f"Probability: {probability:.2%}")
                st.info(
                    "The model predicts that this patient has a low risk of diabetes."
                )

            # Show feature importance if available
            if hasattr(model, "feature_importances_"):
                st.subheader("Feature Importance")

                # Create feature importance chart
                importance = model.feature_importances_
                features = input_df.columns

                # Sort by importance
                indices = np.argsort(importance)
                features = [features[i] for i in indices]
                importance = [importance[i] for i in indices]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.barh(features, importance)
                ax.set_xlabel("Importance")
                ax.set_title("Feature Importance")
                st.pyplot(fig)

# Add explanation section
st.markdown("---")
st.header("About this App")
st.markdown("""
This app uses a machine learning model trained on the UCI Diabetes Dataset to predict the risk of diabetes based on patient information.

**Key factors influencing diabetes risk:**
- Glucose level: High blood sugar is a key indicator of diabetes
- BMI: Being overweight or obese increases risk
- Age: Risk increases with age
- Family history: Captured by Diabetes Pedigree Function
- Blood pressure: Hypertension is associated with diabetes
""")

# Add model monitoring information
st.markdown("---")
st.header("Model Performance Monitoring")

try:
    # Set up MLflow
    if "mlflow" not in st.session_state:
        setup_mlflow()
        st.session_state.mlflow = True

    # Get monitoring runs
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=[
            client.get_experiment_by_name("diabetes-prediction").experiment_id
        ],
        filter_string="tags.run_type = 'monitoring'",
        max_results=5,
    )

    if runs:
        monitoring_data = []

        for run in runs:
            run_data = {
                "Date": run.data.tags.get("monitoring_date", "Unknown"),
                "Accuracy": run.data.metrics.get("current_accuracy", 0),
                "ROC AUC": run.data.metrics.get("current_roc_auc", 0),
                "Drift (%)": run.data.metrics.get("feature_drift_percentage", 0),
            }
            monitoring_data.append(run_data)

        monitoring_df = pd.DataFrame(monitoring_data)
        st.table(monitoring_df)
    else:
        st.info("No monitoring data available yet.")

except Exception as e:
    st.error(f"Error retrieving monitoring data: {e}")

# Footer
st.markdown("---")
st.markdown("**Diabetes Prediction MLOps Project** | Developed using MLflow")
