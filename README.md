# MLFlow Diabetes Classification

This project implements a machine learning-based diabetes prediction system using MLflow for model management and Streamlit for the web interface. The system predicts whether a patient is at risk of diabetes based on various health metrics.

## Features

- Interactive web interface for diabetes risk prediction
- MLflow integration for model tracking and serving
- Real-time predictions using a trained machine learning model
- Input validation for patient health metrics

## Prerequisites

- Python 3.7+
- pip (Python package installer)
- MLflow server running locally

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd MLFlowDiabetesClassification
```

2. Create and activate a virtual environment:
```bash
python -m venv env
source env/bin/activate  # On Windows, use: env\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
MLFlowDiabetesClassification/
├── app.py              # Streamlit web application
├── requirements.txt    # Project dependencies
├── scripts/           # Utility scripts
├── config/           # Configuration files
├── data/             # Dataset directory
└── mlruns/           # MLflow tracking directory
```

## Usage
1. Run the following scripts
```bash
python scripts/preprocess.py
```
```bash
python scripts/train_model.py
```
```bash
python scripts/tune_model.py
```
```bash
python scripts/register_and_transistion.py
```

2. Start the MLflow model server:
```bash
mlflow models serve -m models:/Diabetes_RF_Model/Production -p 1234
```

3. Run the Streamlit application:
```bash
streamlit run app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

4. Enter the patient's health metrics in the web interface:
   - Pregnancies
   - Glucose level
   - Blood pressure
   - Skin thickness
   - Insulin level
   - BMI
   - Diabetes pedigree function
   - Age

5. Click the "Predict" button to get the diabetes risk prediction

## Input Parameters

- **Pregnancies**: Number of times pregnant (0-20)
- **Glucose**: Plasma glucose concentration (0-200 mg/dL)
- **Blood Pressure**: Diastolic blood pressure (0-150 mm Hg)
- **Skin Thickness**: Triceps skin fold thickness (0-100 mm)
- **Insulin**: 2-Hour serum insulin (0-900 mu U/ml)
- **BMI**: Body mass index (0-70 kg/m²)
- **Diabetes Pedigree Function**: Diabetes pedigree function (0-3)
- **Age**: Age in years (0-120)

## Model Information

The prediction model is served through MLflow and provides binary classification results:
- 1: Diabetic
- 0: Non-Diabetic
