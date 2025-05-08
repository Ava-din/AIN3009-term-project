"""
Evaluate trained models on test data and log results to MLflow.
"""

import os
import sys
import logging
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TARGET_COLUMN, MODEL_NAME
from utils.mlflow_utils import setup_mlflow, promote_model, get_latest_model_version

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_test_data():
    """Load the test dataset."""
    test_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(test_path):
        logger.error(f"Test data not found at {test_path}. Run preprocess.py first.")
        return None, None

    logger.info(f"Loading test data from {test_path}")
    df = pd.read_csv(test_path)

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    logger.info(f"Test data loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def load_model(run_id=None, model_version=None):
    """Load a model from MLflow."""
    try:
        if run_id:
            # Load from a specific run
            model_uri = f"runs:/{run_id}/model"
            logger.info(f"Loading model from run: {run_id}")
        elif model_version:
            # Load from model registry
            model_uri = f"models:/{MODEL_NAME}/{model_version}"
            logger.info(
                f"Loading model from registry: {MODEL_NAME} version {model_version}"
            )
        else:
            # Load the latest model from the registry
            latest_version = get_latest_model_version(MODEL_NAME)
            if latest_version is None:
                logger.error("No model versions found in registry")
                return None
            model_uri = f"models:/{MODEL_NAME}/{latest_version}"
            logger.info(
                f"Loading latest model from registry: {MODEL_NAME} version {latest_version}"
            )

        model = mlflow.sklearn.load_model(model_uri)
        return model

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def evaluate_model(model, X, y):
    """Evaluate a model on test data."""
    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)

    # Calculate metrics
    metrics = {}
    metrics["accuracy"] = accuracy_score(y, y_pred)
    metrics["precision"] = precision_score(y, y_pred, average="weighted")
    metrics["recall"] = recall_score(y, y_pred, average="weighted")
    metrics["f1_score"] = f1_score(y, y_pred, average="weighted")

    # For binary classification
    if y_prob.shape[1] == 2:
        metrics["roc_auc"] = roc_auc_score(y, y_prob[:, 1])
    else:
        metrics["roc_auc"] = roc_auc_score(
            y, y_prob, multi_class="ovr", average="weighted"
        )

    # Log metrics
    logger.info("Model evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # Create classification report
    class_report = classification_report(y, y_pred)
    logger.info(f"Classification report:\n{class_report}")

    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    logger.info(f"Confusion matrix:\n{cm}")

    return metrics, y_pred, y_prob


def create_evaluation_plots(y_true, y_pred, y_prob):
    """Create evaluation plots and save them as files."""
    try:
        # Create a directory for plots if it doesn't exist
        plots_dir = os.path.join(os.getcwd(), "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # For binary classification
        if y_prob.shape[1] == 2:
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(
                fpr,
                tpr,
                label=f"ROC curve (AUC = {roc_auc_score(y_true, y_prob[:, 1]):.3f})",
            )
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend(loc="lower right")
            roc_path = os.path.join(plots_dir, "roc_curve.png")
            plt.savefig(roc_path)
            plt.close()

            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, label="Precision-Recall curve")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Precision-Recall Curve")
            plt.legend(loc="lower left")
            pr_path = os.path.join(plots_dir, "precision_recall_curve.png")
            plt.savefig(pr_path)
            plt.close()

        # Confusion matrix heatmap
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        cm_path = os.path.join(plots_dir, "confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close()

        logger.info(f"Evaluation plots saved to {plots_dir}")
        return {
            "roc_curve": roc_path if y_prob.shape[1] == 2 else None,
            "pr_curve": pr_path if y_prob.shape[1] == 2 else None,
            "confusion_matrix": cm_path,
        }

    except Exception as e:
        logger.error(f"Error creating evaluation plots: {e}")
        return {}


def log_evaluation_to_mlflow(metrics, plot_paths, model_version):
    """Log evaluation metrics and plots to MLflow."""
    try:
        # Start a new run for evaluation
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"Started MLflow run for evaluation: {run_id}")

            # Log metrics
            for name, value in metrics.items():
                mlflow.log_metric(name, value)

            # Log plots as artifacts
            for plot_name, plot_path in plot_paths.items():
                if plot_path:
                    mlflow.log_artifact(plot_path)

            # Log the model version being evaluated
            mlflow.log_param("model_version", model_version)

            # Tag the run for easier filtering
            mlflow.set_tag("run_type", "evaluation")

            logger.info(f"Evaluation logged to MLflow run: {run_id}")
            return run_id

    except Exception as e:
        logger.error(f"Error logging evaluation to MLflow: {e}")
        return None


def evaluate_and_promote_best_model():
    """Evaluate all model versions and promote the best to production."""
    # Set up MLflow
    setup_mlflow()

    # Load test data
    X_test, y_test = load_test_data()
    if X_test is None or y_test is None:
        return

    # Get the MLflow client
    client = mlflow.tracking.MlflowClient()

    # Get all model versions
    model_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not model_versions:
        logger.error(f"No model versions found for {MODEL_NAME}")
        return

    # Evaluate each model version
    results = []
    for mv in model_versions:
        version = mv.version
        current_stage = mv.current_stage

        logger.info(
            f"Evaluating model {MODEL_NAME} version {version} (current stage: {current_stage})"
        )

        # Load the model
        model = load_model(model_version=version)
        if model is None:
            continue

        # Evaluate the model
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        # Create evaluation plots
        plot_paths = create_evaluation_plots(y_test, y_pred, y_prob)

        # Log evaluation to MLflow
        run_id = log_evaluation_to_mlflow(metrics, plot_paths, version)

        # Store results
        results.append(
            {
                "version": version,
                "current_stage": current_stage,
                "metrics": metrics,
                "run_id": run_id,
            }
        )

    # Find the best model version based on AUC
    if results:
        best_model = max(results, key=lambda x: x["metrics"]["roc_auc"])
        logger.info(
            f"Best model: version {best_model['version']} with ROC AUC: {best_model['metrics']['roc_auc']:.4f}"
        )

        # Promote to production if not already
        if best_model["current_stage"] != "Production":
            promote_model(MODEL_NAME, best_model["version"], "Production")
            logger.info(
                f"Model {MODEL_NAME} version {best_model['version']} promoted to Production"
            )
        else:
            logger.info(
                f"Model {MODEL_NAME} version {best_model['version']} is already in Production"
            )

        return best_model

    return None


def main():
    """Main function to evaluate models."""
    evaluate_and_promote_best_model()


if __name__ == "__main__":
    main()
