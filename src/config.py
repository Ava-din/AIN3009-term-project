"""
Configuration settings for the diabetes prediction MLOps project.
"""

import os
from pathlib import Path

# Project paths
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
MLRUNS_DIR = ROOT_DIR / "mlruns"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# Data settings
RAW_DATA_FILE = DATA_DIR / "diabetes.csv"
PROCESSED_DATA_FILE = DATA_DIR / "processed_diabetes.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_COLUMN = "Outcome"

# MLflow settings
MLFLOW_TRACKING_URI = "file://" + str(MLRUNS_DIR.absolute())
EXPERIMENT_NAME = "diabetes-prediction"

# Model training settings
CV_FOLDS = 5
SCORING = "roc_auc"

# Model registry settings
MODEL_NAME = "diabetes-classifier"

# Hyperparameter tuning
N_ITER = 50

# Model monitoring settings
DRIFT_THRESHOLD = 0.1
