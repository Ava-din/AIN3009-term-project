"""
Monitor model performance and detect data drift over time.
"""

import os
import sys
import logging
import mlflow
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import joblib

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, TARGET_COLUMN, MODEL_NAME, DRIFT_THRESHOLD
from utils.mlflow_utils import setup_mlflow, get_latest_model_version

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_reference_data():
    """Load the original training data as a reference."""
    reference_path = os.path.join(DATA_DIR, "train.csv")

    if not os.path.exists(reference_path):
        logger.error(
            f"Reference data not found at {reference_path}. Run preprocess.py first."
        )
        return None

    logger.info(f"Loading reference data from {reference_path}")
    reference_data = pd.read_csv(reference_path)

    return reference_data


def load_current_data():
    """
    Load current data for monitoring.
    In a real production system, this would be new incoming data.
    For demonstration, we'll use the test set.
    """
    current_path = os.path.join(DATA_DIR, "test.csv")

    if not os.path.exists(current_path):
        logger.error(
            f"Current data not found at {current_path}. Run preprocess.py first."
        )
        return None

    logger.info(f"Loading current data from {current_path}")
    current_data = pd.read_csv(current_path)

    return current_data


def detect_feature_drift(reference_df, current_df):
    """
    Detect drift in feature distributions using Kolmogorov-Smirnov test.
    """
    drift_results = {}

    # Get common columns excluding the target
    columns = [col for col in reference_df.columns if col != TARGET_COLUMN]

    # Calculate drift for each feature
    for col in columns:
        # Extract feature values
        ref_values = reference_df[col].values
        curr_values = current_df[col].values

        # Skip non-numeric features
        if not np.issubdtype(ref_values.dtype, np.number):
            continue

        # Perform KS test
        ks_stat, p_value = ks_2samp(ref_values, curr_values)

        # Determine if drift detected based on p-value
        drift_detected = p_value < 0.05

        drift_results[col] = {
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "drift_detected": drift_detected,
        }

    return drift_results


def detect_target_drift(reference_df, current_df):
    """
    Detect drift in target variable distribution.
    For classification problems, compare class distributions.
    """
    if (
        TARGET_COLUMN not in reference_df.columns
        or TARGET_COLUMN not in current_df.columns
    ):
        logger.error(f"Target column '{TARGET_COLUMN}' not found in data")
        return None

    # Get target distributions
    ref_target = reference_df[TARGET_COLUMN].value_counts(normalize=True)
    curr_target = current_df[TARGET_COLUMN].value_counts(normalize=True)

    # Ensure both distributions have the same classes
    all_classes = sorted(set(ref_target.index) | set(curr_target.index))
    ref_dist = {cls: ref_target.get(cls, 0) for cls in all_classes}
    curr_dist = {cls: curr_target.get(cls, 0) for cls in all_classes}

    # Calculate Jensen-Shannon divergence or a simpler metric like absolute difference
    js_divergence = 0
    for cls in all_classes:
        abs_diff = abs(ref_dist[cls] - curr_dist[cls])
        js_divergence += abs_diff

    js_divergence /= 2  # Normalize to range [0, 1]

    # Determine if drift detected
    drift_detected = js_divergence > DRIFT_THRESHOLD

    result = {
        "divergence": js_divergence,
        "ref_distribution": ref_dist,
        "current_distribution": curr_dist,
        "drift_detected": drift_detected,
    }

    return result


def evaluate_model_on_current_data(current_df):
    """
    Evaluate the production model on current data and record performance.
    """
    try:
        # Get the latest production model
        client = mlflow.tracking.MlflowClient()
        production_models = client.get_latest_versions(
            MODEL_NAME, stages=["Production"]
        )

        if not production_models:
            logger.error(f"No production model found for {MODEL_NAME}")
            return None

        # Load the model
        model_uri = f"models:/{MODEL_NAME}/{production_models[0].version}"
        model = mlflow.sklearn.load_model(model_uri)

        # Prepare data
        X = current_df.drop(TARGET_COLUMN, axis=1)
        y = current_df[TARGET_COLUMN]

        # Make predictions
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X) if hasattr(model, "predict_proba") else None

        # Calculate metrics
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        metrics = {}
        metrics["accuracy"] = accuracy_score(y, y_pred)
        metrics["precision"] = precision_score(y, y_pred, average="weighted")
        metrics["recall"] = recall_score(y, y_pred, average="weighted")
        metrics["f1_score"] = f1_score(y, y_pred, average="weighted")

        if y_prob is not None:
            if y_prob.shape[1] == 2:  # Binary classification
                metrics["roc_auc"] = roc_auc_score(y, y_prob[:, 1])
            else:  # Multi-class
                metrics["roc_auc"] = roc_auc_score(
                    y, y_prob, multi_class="ovr", average="weighted"
                )

        return metrics

    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return None


def plot_drift_metrics(drift_results, save_dir=None):
    """
    Create plots for drift metrics.
    """
    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), "drift_plots")

    os.makedirs(save_dir, exist_ok=True)

    # Create feature drift plot
    plt.figure(figsize=(12, 8))
    features = list(drift_results.keys())
    ks_stats = [drift_results[feat]["ks_statistic"] for feat in features]
    drift_detected = [drift_results[feat]["drift_detected"] for feat in features]

    colors = ["red" if d else "green" for d in drift_detected]

    plt.barh(features, ks_stats, color=colors)
    plt.axvline(
        x=DRIFT_THRESHOLD,
        color="red",
        linestyle="--",
        label=f"Threshold ({DRIFT_THRESHOLD})",
    )
    plt.xlabel("KS Statistic")
    plt.title("Feature Drift Detection")
    plt.legend()

    plot_path = os.path.join(save_dir, "feature_drift.png")
    plt.savefig(plot_path)
    plt.close()

    logger.info(f"Drift plots saved to {save_dir}")

    return {"feature_drift_plot": plot_path}


def log_monitoring_to_mlflow(feature_drift, target_drift, model_metrics, plot_paths):
    """
    Log monitoring results to MLflow.
    """
    # Start a new run
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow run for monitoring: {run_id}")

        # Log model metrics
        if model_metrics:
            for name, value in model_metrics.items():
                mlflow.log_metric(f"current_{name}", value)

        # Log drift metrics
        drifted_features = sum(1 for f in feature_drift.values() if f["drift_detected"])
        total_features = len(feature_drift)
        drift_percentage = (
            (drifted_features / total_features) * 100 if total_features > 0 else 0
        )

        mlflow.log_metric("feature_drift_percentage", drift_percentage)

        if target_drift:
            mlflow.log_metric("target_drift_divergence", target_drift["divergence"])
            mlflow.log_param("target_drift_detected", target_drift["drift_detected"])

        # Log plots
        for plot_name, plot_path in plot_paths.items():
            mlflow.log_artifact(plot_path)

        # Set tags
        mlflow.set_tag("run_type", "monitoring")
        mlflow.set_tag("monitoring_date", datetime.now().strftime("%Y-%m-%d"))

        logger.info(f"Monitoring logged to MLflow run: {run_id}")
        return run_id


def monitor_model_performance():
    """
    Main function to monitor model performance and detect data drift.
    """
    # Set up MLflow
    setup_mlflow()

    # Load reference and current data
    reference_df = load_reference_data()
    current_df = load_current_data()

    if reference_df is None or current_df is None:
        return

    # Detect feature drift
    logger.info("Detecting feature drift...")
    feature_drift = detect_feature_drift(reference_df, current_df)

    # Log feature drift results
    drifted_features = [f for f, res in feature_drift.items() if res["drift_detected"]]
    if drifted_features:
        logger.warning(
            f"Drift detected in {len(drifted_features)} features: {drifted_features}"
        )
    else:
        logger.info("No feature drift detected")

    # Detect target drift
    logger.info("Detecting target drift...")
    target_drift = detect_target_drift(reference_df, current_df)

    if target_drift and target_drift["drift_detected"]:
        logger.warning(
            f"Target drift detected! Divergence: {target_drift['divergence']:.4f}"
        )
    else:
        logger.info("No target drift detected")

    # Evaluate model on current data
    logger.info("Evaluating model on current data...")
    model_metrics = evaluate_model_on_current_data(current_df)

    if model_metrics:
        logger.info("Current model performance:")
        for metric, value in model_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

    # Create drift plots
    plot_paths = plot_drift_metrics(feature_drift)

    # Log to MLflow
    logger.info("Logging monitoring results to MLflow...")
    run_id = log_monitoring_to_mlflow(
        feature_drift, target_drift, model_metrics, plot_paths
    )

    # Determine if retraining is needed
    needs_retraining = (
        drifted_features and len(drifted_features) / len(feature_drift) > 0.3
    ) or (target_drift and target_drift["drift_detected"])

    if needs_retraining:
        logger.warning("Model retraining recommended due to significant data drift")
    else:
        logger.info("No retraining needed at this time")


if __name__ == "__main__":
    monitor_model_performance()
