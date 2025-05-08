#!/usr/bin/env python
"""
Quickstart Guide for Diabetes Prediction MLOps Project
======================================================

This script provides a guide to the MLOps project structure and how to run each component.
You can also run this script to execute individual components of the pipeline or the full pipeline.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 80)
    print(f" {title} ".center(80, "="))
    print("=" * 80 + "\n")


def print_section(title):
    """Print a section title."""
    print("\n" + "-" * 80)
    print(f" {title} ".center(80, "-"))
    print("-" * 80 + "\n")


def execute_command(command):
    """Execute a shell command and print output."""
    print(f"Running: {command}\n")
    process = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(process.stdout)
    if process.returncode != 0:
        print(f"Error: {process.stderr}")
    return process.returncode


def project_overview():
    """Print the project overview."""
    print_header("Diabetes Prediction MLOps Project")

    print("""
This project implements a complete machine learning pipeline for diabetes prediction 
using the UCI Diabetes Dataset with MLflow as the MLOps platform.

The pipeline includes:
1. Data acquisition and preprocessing
2. Model training and hyperparameter optimization
3. Model evaluation and selection
4. Model deployment
5. Model monitoring and drift detection

All these stages are managed using MLflow, an open-source platform for managing the ML lifecycle.
    """)


def dataset_description():
    """Print the dataset description."""
    print_section("Dataset Description")

    print("""
The UCI Diabetes Dataset contains information about female patients of Pima Indian heritage 
with the following features:

- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skinfold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)²)
- DiabetesPedigreeFunction: Diabetes pedigree function (score based on family history)
- Age: Age in years
- Outcome: Class variable (0 or 1) indicating presence of diabetes (1 = has diabetes)
    """)


def project_structure():
    """Print the project structure."""
    print_section("Project Structure")

    print("""
mlops-project/
├── app/                   # Web applications for model serving
│   ├── model/             # Deployed model files
│   ├── app.py             # Flask API
│   └── streamlit_app.py   # Streamlit interactive app
├── data/                  # Data files
│   ├── diabetes.csv       # Raw data
│   ├── processed_diabetes.csv # Processed data
│   ├── train.csv          # Training data
│   └── test.csv           # Test data
├── models/                # Saved model artifacts
├── mlruns/                # MLflow experiment tracking data
├── notebooks/             # Jupyter notebooks
│   └── quickstart.py      # This script
├── src/                   # Source code
│   ├── data/              # Data processing scripts
│   ├── models/            # Model training scripts
│   ├── evaluation/        # Model evaluation scripts
│   ├── monitoring/        # Model monitoring scripts
│   ├── utils/             # Utility functions
│   └── config.py          # Configuration settings
├── .gitignore             # Git ignore file
├── README.md              # Project README
├── requirements.txt       # Dependencies
└── run_pipeline.py        # Main pipeline script
    """)


def setup_instructions():
    """Print setup instructions."""
    print_section("Environment Setup")

    print("""
1. Create a virtual environment:
   $ python -m venv venv
   $ source venv/bin/activate  # On Windows: venv\\Scripts\\activate

2. Install dependencies:
   $ pip install -r requirements.txt
    """)


def pipeline_components():
    """Print information about pipeline components."""
    print_section("Pipeline Components")

    components = [
        {
            "name": "Data Download",
            "description": "Downloads the UCI Diabetes Dataset",
            "command": "python src/data/download_data.py",
        },
        {
            "name": "Data Preprocessing",
            "description": "Cleans data, handles missing values, and creates features",
            "command": "python src/data/preprocess.py",
        },
        {
            "name": "Model Training",
            "description": "Trains multiple model types with basic parameters",
            "command": "python src/models/train.py",
        },
        {
            "name": "Hyperparameter Tuning",
            "description": "Optimizes model hyperparameters using Hyperopt",
            "command": "python src/models/hyperparameter_tuning.py",
        },
        {
            "name": "Model Evaluation",
            "description": "Evaluates models on test data and promotes the best to production",
            "command": "python src/evaluation/evaluate.py",
        },
        {
            "name": "Model Deployment",
            "description": "Deploys the production model for inference",
            "command": "python src/models/deploy.py",
        },
        {
            "name": "Model Monitoring",
            "description": "Monitors model performance and detects data drift",
            "command": "python src/monitoring/monitor.py",
        },
    ]

    for i, component in enumerate(components, 1):
        print(f"{i}. {component['name']}")
        print(f"   Description: {component['description']}")
        print(f"   Command: {component['command']}")
        print()


def mlops_benefits():
    """Print MLOps benefits."""
    print_section("MLOps Benefits and Best Practices")

    print("""
This project demonstrates several MLOps best practices:

1. Experiment Tracking: All experiments are tracked in MLflow, including parameters, metrics, 
   and artifacts.
2. Model Registry: Models are versioned and managed through the MLflow Model Registry.
3. Reproducibility: The entire pipeline is reproducible through scripts and version control.
4. Continuous Evaluation: Models are continuously evaluated on new data to detect drift.
5. Automated Deployment: The best model is automatically deployed for inference.

These practices help ensure that machine learning models in healthcare can be developed, 
validated, and deployed reliably and efficiently.
    """)


def usage_examples():
    """Print usage examples."""
    print_section("Usage Examples")

    print("""
1. Run the full pipeline:
   $ python run_pipeline.py

2. View MLflow experiments:
   $ mlflow ui

3. Run the Streamlit app:
   $ streamlit run app/streamlit_app.py

4. Use the Flask API:
   $ python app/app.py
   
5. Run this script with a specific component:
   $ python notebooks/quickstart.py --download-data
   $ python notebooks/quickstart.py --train-models
   $ python notebooks/quickstart.py --deploy
    """)


def main():
    """Main function to run the quickstart guide."""
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Diabetes Prediction MLOps Project Quickstart Guide"
    )

    # Add arguments for each pipeline component
    parser.add_argument(
        "--download-data", action="store_true", help="Download the dataset"
    )
    parser.add_argument("--preprocess", action="store_true", help="Preprocess the data")
    parser.add_argument("--train-models", action="store_true", help="Train models")
    parser.add_argument(
        "--tune-hyperparams", action="store_true", help="Tune hyperparameters"
    )
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--deploy", action="store_true", help="Deploy the model")
    parser.add_argument(
        "--monitor", action="store_true", help="Monitor model performance"
    )
    parser.add_argument(
        "--run-pipeline", action="store_true", help="Run the full pipeline"
    )
    parser.add_argument(
        "--guide", action="store_true", help="Show the quickstart guide"
    )

    # Parse arguments
    args = parser.parse_args()

    # If no arguments provided or guide requested, show the full guide
    if len(sys.argv) == 1 or args.guide:
        project_overview()
        dataset_description()
        project_structure()
        setup_instructions()
        pipeline_components()
        mlops_benefits()
        usage_examples()
        return

    # Run requested components
    if args.download_data:
        print_section("Downloading Data")
        execute_command("python src/data/download_data.py")

    if args.preprocess:
        print_section("Preprocessing Data")
        execute_command("python src/data/preprocess.py")

    if args.train_models:
        print_section("Training Models")
        execute_command("python src/models/train.py")

    if args.tune_hyperparams:
        print_section("Tuning Hyperparameters")
        execute_command("python src/models/hyperparameter_tuning.py")

    if args.evaluate:
        print_section("Evaluating Models")
        execute_command("python src/evaluation/evaluate.py")

    if args.deploy:
        print_section("Deploying Model")
        execute_command("python src/models/deploy.py")

    if args.monitor:
        print_section("Monitoring Model")
        execute_command("python src/monitoring/monitor.py")

    if args.run_pipeline:
        print_section("Running Full Pipeline")
        execute_command("python run_pipeline.py")


if __name__ == "__main__":
    main()
