# Technical Documentation - MLFlow Diabetes Classification

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Processing](#data-processing)
3. [Model Development](#model-development)
4. [MLflow Integration](#mlflow-integration)
5. [API Endpoints](#api-endpoints)
6. [Web Application](#web-application)
7. [Deployment](#deployment)
8. [Performance Metrics](#performance-metrics)
9. [Troubleshooting](#troubleshooting)

## System Architecture

### Overview
The system follows a microservices architecture with three main components:
1. MLflow Model Server (Prediction Service)
2. Streamlit Web Interface (Frontend)
3. Model Training Pipeline (Offline Component)

### Component Interaction
```
[Streamlit Frontend] <--HTTP--> [MLflow Model Server] <--Filesystem--> [MLflow Model Registry]
         ↑                              ↑                                      ↑
         |                              |                                      |
    User Interface               REST API (Port 1234)                  Trained Models
```

## Data Processing

### Dataset Structure
The diabetes prediction model uses the Pima Indians Diabetes Database with the following features:

| Feature                    | Type    | Range        | Description                                      |
|---------------------------|---------|--------------|--------------------------------------------------|
| Pregnancies               | Integer | 0-20         | Number of times pregnant                         |
| Glucose                   | Integer | 0-200        | Plasma glucose concentration                     |
| BloodPressure            | Integer | 0-150        | Diastolic blood pressure (mm Hg)                |
| SkinThickness            | Integer | 0-100        | Triceps skin fold thickness (mm)                |
| Insulin                   | Integer | 0-900        | 2-Hour serum insulin (mu U/ml)                  |
| BMI                       | Float   | 0.0-70.0     | Body mass index (kg/m²)                         |
| DiabetesPedigreeFunction | Float   | 0.0-3.0      | Diabetes pedigree function                      |
| Age                       | Integer | 0-120        | Age in years                                    |

### Data Preprocessing Steps
1. Missing Value Treatment
   - Replace zeros with median values for Glucose, BloodPressure, SkinThickness, Insulin, and BMI
   - Handle outliers using IQR method

2. Feature Scaling
   - StandardScaler for numerical features
   - No encoding needed (all features are numerical)

## Model Development

### Model Architecture
The classification model pipeline includes:
1. Data preprocessing steps
2. Feature scaling
3. Classification algorithm
4. Hyperparameter optimization

### Training Process
```python
# Example training pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', selected_classifier)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)
```

## MLflow Integration

### Model Tracking
MLflow tracks the following metrics and parameters:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Training parameters
- Model hyperparameters

### Model Registry
Models are registered using MLflow's model registry:
```bash
mlflow models serve -m "models:/diabetes_classifier/production" -p 1234
```

### Experiment Tracking
Each training run is logged with:
- Parameters
- Metrics
- Model artifacts
- Environment details

## API Endpoints

### Prediction Endpoint
- **URL**: `http://localhost:1234/invocations`
- **Method**: POST
- **Content-Type**: application/json
- **Request Format**:
```json
{
    "inputs": [
        [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]
    ]
}
```
- **Response Format**:
```json
{
    "predictions": [0 or 1]
}
```

### Error Handling
The API implements standard HTTP status codes:
- 200: Successful prediction
- 400: Invalid input data
- 500: Server error

## Web Application

### Streamlit Components
The web interface is built using Streamlit with the following components:
1. Input validation
2. Real-time prediction
3. Error handling
4. Result display

### Input Validation Rules
```python
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20)
glucose = st.number_input("Glucose", min_value=0, max_value=200)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150)
...
```

## Deployment

### Local Deployment
1. Start MLflow server:
```bash
mlflow models serve -m <path-to-model> -p 1234
```

2. Launch Streamlit app:
```bash
streamlit run app.py
```

### Production Deployment Considerations
1. Use production-grade WSGI server
2. Implement load balancing
3. Set up monitoring and logging
4. Configure security measures
5. Use environment variables for sensitive data

## Performance Metrics

### Model Evaluation
The model's performance is evaluated using:
- Cross-validation scores
- Confusion matrix
- ROC curve
- Precision-Recall curve

### System Performance
Monitor the following metrics:
- Response time
- Request throughput
- Error rate
- Resource utilization

## Troubleshooting

### Common Issues and Solutions

1. MLflow Server Connection Issues
   - Check if MLflow server is running
   - Verify port 1234 is available
   - Check network connectivity

2. Model Loading Errors
   - Verify model path
   - Check MLflow environment
   - Validate model version

3. Prediction Errors
   - Validate input data format
   - Check feature scaling
   - Verify model serving status

### Logging and Monitoring
- Application logs location: `logs/app.log`
- MLflow tracking URI: `mlruns/`
- System metrics: Prometheus/Grafana (if configured)

## Future Improvements

1. Model Enhancements
   - Feature engineering
   - Ensemble methods
   - Deep learning integration

2. System Improvements
   - Automated retraining
   - A/B testing
   - Batch prediction support
   - Real-time monitoring dashboard 