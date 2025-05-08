# Diabetes Prediction MLOps Project

A machine learning operations (MLOps) project for diabetes prediction based on the UCI Diabetes Dataset.

## Project Overview

This project implements a complete MLOps pipeline for diabetes prediction, including:

- Data acquisition and preprocessing
- Model training and hyperparameter tuning
- Model evaluation and deployment
- Model monitoring and drift detection
- Interactive web application for predictions

## Directory Structure

```
├── app                     # Streamlit web application
├── data                    # Data directory
├── drift_plots             # Data drift monitoring plots
├── mlruns                  # MLflow tracking data
├── models                  # Saved model files
├── notebooks               # Jupyter notebooks for exploration
├── src                     # Source code
│   ├── data                # Data processing scripts
│   ├── evaluation          # Model evaluation scripts
│   ├── models              # Model training and deployment
│   ├── monitoring          # Model monitoring and drift detection
│   └── utils               # Utility functions
├── run_pipeline.py         # Main pipeline runner
└── requirements.txt        # Project dependencies
```

## Setup and Installation

1. Clone the repository:

```bash
git clone https://github.com/Ava-din/AIN3009-term-project.git
cd mlops-project
```

2. Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Pipeline

The entire MLOps pipeline can be executed with a single command:

```bash
python run_pipeline.py
```

This will run the following steps sequentially:
1. Download the diabetes dataset
2. Preprocess the data
3. Train multiple models
4. Perform hyperparameter tuning
5. Evaluate model performance
6. Deploy the best model to production
7. Set up model monitoring

## Web Application

The project includes a Streamlit web application for making predictions:

```bash
streamlit run app/streamlit_app.py
```

The app allows users to:
- Input patient information
- Get diabetes risk predictions
- View feature importance
- Monitor model performance over time

## MLflow Tracking

All experiments are tracked using MLflow:

```bash
mlflow ui
```

This will start the MLflow UI where you can:
- Compare model performance
- View hyperparameter configurations
- See evaluation metrics
- Manage model versions

## Model Monitoring

The project includes a monitoring system that:
- Detects data drift
- Tracks model performance over time
- Generates alerts when performance degrades
- Creates visualizations of model and data health

## Project Components

### Data Processing

- `download_data.py`: Downloads the UCI Diabetes Dataset
- `preprocess.py`: Cleans and transforms the data, creates train/test splits

### Model Training

- `train.py`: Trains multiple models (Random Forest, Gradient Boosting, etc.)
- `hyperparameter_tuning.py`: Uses Hyperopt for hyperparameter optimization

### Model Evaluation

- Evaluates models on test data
- Calculates metrics like accuracy, ROC AUC, precision, recall
- Generates performance reports

### Model Deployment

- `deploy.py`: Registers the best model in MLflow Model Registry
- Transitions models through staging to production

### Model Monitoring

- `monitor.py`: Sets up continuous monitoring
- Detects data drift and model performance degradation
- Generates alerts and visualizations

## Configuration

The project configuration is in `src/config.py`, which includes:
- File paths
- MLflow settings
- Model training parameters
- Monitoring thresholds

## Tech Stack

- **Python**: Core programming language
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data processing
- **MLflow**: Experiment tracking and model registry
- **Hyperopt**: Hyperparameter optimization
- **Streamlit**: Web application
- **Matplotlib & Seaborn**: Visualization