#!/usr/bin/env python
"""
Main script to run the entire MLOps pipeline for diabetes prediction.
"""

import os
import sys
import subprocess
import logging
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pipeline.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_step(step_name, command):
    """Run a pipeline step and log the output."""
    logger.info(f"Starting step: {step_name}")
    start_time = time.time()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            shell=True,
        )

        # Stream and log output in real-time
        for line in process.stdout:
            line = line.strip()
            if line:
                logger.info(f"{step_name}: {line}")

        # Wait for process to complete
        process.wait()

        # Check for errors
        if process.returncode != 0:
            error_output = process.stderr.read()
            logger.error(f"{step_name} failed with error code {process.returncode}")
            logger.error(f"Error output: {error_output}")
            return False

        duration = time.time() - start_time
        logger.info(f"Completed step: {step_name} in {duration:.2f} seconds")
        return True

    except Exception as e:
        logger.error(f"Error running {step_name}: {e}")
        return False


def run_pipeline():
    """Run the full MLOps pipeline."""
    pipeline_start = time.time()
    logger.info("Starting MLOps pipeline run")

    # Define pipeline steps and their commands
    pipeline_steps = [
        {"name": "Download Data", "command": "python src/data/download_data.py"},
        {"name": "Preprocess Data", "command": "python src/data/preprocess.py"},
        {"name": "Train Models", "command": "python src/models/train.py"},
        {
            "name": "Hyperparameter Tuning",
            "command": "python src/models/hyperparameter_tuning.py",
        },
        {"name": "Evaluate Models", "command": "python src/evaluation/evaluate.py"},
        {"name": "Deploy Model", "command": "python src/models/deploy.py"},
        {"name": "Monitor Performance", "command": "python src/monitoring/monitor.py"},
    ]

    # Run each step in sequence
    success = True
    for step in pipeline_steps:
        success = run_step(step["name"], step["command"])
        if not success:
            logger.error(f"Pipeline failed at step: {step['name']}")
            break

    # Calculate total pipeline duration
    pipeline_duration = time.time() - pipeline_start

    if success:
        logger.info(
            f"MLOps pipeline completed successfully in {pipeline_duration:.2f} seconds"
        )

        # Start the Streamlit app
        logger.info("Starting Streamlit app")
        print("\nTo start the Streamlit app, run the following command:")
        print("streamlit run app/streamlit_app.py")

        # Start MLflow UI
        logger.info("Starting MLflow UI")
        print("\nTo view MLflow experiments, run the following command:")
        print("mlflow ui")
    else:
        logger.error(f"MLOps pipeline failed after {pipeline_duration:.2f} seconds")


if __name__ == "__main__":
    run_pipeline()
