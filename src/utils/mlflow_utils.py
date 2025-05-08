"""
Utility functions for MLflow tracking and model registry.
"""

import os
import sys
import mlflow
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np
import pandas as pd

# Add the parent directory to the path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MLFLOW_TRACKING_URI, EXPERIMENT_NAME, MODEL_NAME

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Set up MLflow tracking server and experiment."""
    # Set the tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Create or get the experiment
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
        logger.info(
            f"Created new experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}"
        )
    else:
        experiment_id = experiment.experiment_id
        logger.info(
            f"Using existing experiment '{EXPERIMENT_NAME}' with ID: {experiment_id}"
        )

    # Set the experiment
    mlflow.set_experiment(EXPERIMENT_NAME)

    return experiment_id


def log_model_metrics(y_true, y_pred, y_prob=None):
    """Log model metrics to MLflow."""
    metrics = {}

    # Calculate standard metrics
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average="weighted")
    metrics["recall"] = recall_score(y_true, y_pred, average="weighted")
    metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted")

    # If probability predictions are available, calculate AUC
    if y_prob is not None:
        # For binary classification
        if y_prob.shape[1] == 2:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob[:, 1])
        # For multi-class classification
        else:
            metrics["roc_auc"] = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="weighted"
            )

    # Log metrics to MLflow
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    logger.info(f"Logged metrics: {metrics}")
    return metrics


def register_model(model_uri, run_id):
    """Register the model in the MLflow Model Registry."""
    try:
        # Register the model
        registered_model = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model", name=MODEL_NAME
        )

        logger.info(
            f"Model registered: {registered_model.name} version {registered_model.version}"
        )
        return registered_model

    except Exception as e:
        logger.error(f"Error registering model: {e}")
        return None


def promote_model(model_name, version, stage):
    """Promote a model version to a different stage (Staging, Production, Archived)."""
    try:
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=version, stage=stage
        )

        logger.info(f"Model {model_name} version {version} promoted to {stage}")
        return True

    except Exception as e:
        logger.error(f"Error promoting model: {e}")
        return False


def log_model_explanation(feature_importance, feature_names):
    """Log feature importance as a model explanation artifact."""
    try:
        # Create a dataframe with feature importances
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importance}
        )

        # Sort by importance
        importance_df = importance_df.sort_values("Importance", ascending=False)

        # Save to CSV
        importance_path = "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)

        # Log as an artifact
        mlflow.log_artifact(importance_path)

        # Clean up
        os.remove(importance_path)

        logger.info(f"Logged feature importance for {len(feature_names)} features")
        return True

    except Exception as e:
        logger.error(f"Error logging model explanation: {e}")
        return False


def get_latest_model_version(model_name):
    """Get the latest version of a registered model."""
    try:
        client = mlflow.tracking.MlflowClient()
        model_versions = client.get_latest_versions(model_name)

        if model_versions:
            latest_version = max([int(mv.version) for mv in model_versions])
            logger.info(f"Latest version of {model_name}: {latest_version}")
            return latest_version
        else:
            logger.info(f"No versions found for model {model_name}")
            return None

    except Exception as e:
        logger.error(f"Error retrieving latest model version: {e}")
        return None
