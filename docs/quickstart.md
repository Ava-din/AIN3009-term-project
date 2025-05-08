# Diabetes Prediction MLOps Project: Quick Start Guide

This guide will help you quickly get started with the Diabetes Prediction MLOps project.

## Prerequisites

- Python 3.8+ installed
- Git installed
- Basic knowledge of machine learning concepts

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Ava-din/AIN3009-term-project.git
cd mlops-project
```

2. Create and activate a virtual environment:

```bash
# On Linux/Mac
python -m venv env
source env/bin/activate

# On Windows
python -m venv env
env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Full Pipeline

To run the entire MLOps pipeline (data download, preprocessing, model training, etc.):

```bash
python run_pipeline.py
```

This will take several minutes to complete, depending on your system.

## Using the Web Application

1. Start the Streamlit web application:

```bash
streamlit run app/streamlit_app.py
```

2. Open your browser and navigate to http://localhost:8501

3. Enter patient information in the form and click "Predict Diabetes Risk"

## Viewing MLflow Experiments

1. Start the MLflow UI:

```bash
mlflow ui
```

2. Open your browser and navigate to http://localhost:5000

3. Browse experiments, compare models, and view metrics

## Individual Components

If you want to run specific parts of the pipeline:

### Data Processing Only

```bash
python src/data/download_data.py
python src/data/preprocess.py
```

### Model Training Only

```bash
python src/models/train.py
```

### Hyperparameter Tuning Only

```bash
python src/models/hyperparameter_tuning.py
```

### Model Evaluation Only

```bash
python src/evaluation/evaluate.py
```

### Model Monitoring Only

```bash
python src/monitoring/monitor.py
```

## Troubleshooting

### Common Issues

1. **MLflow database locked**
   - Stop any running MLflow processes
   - Delete the `mlruns/.trash` directory

2. **Missing dependencies**
   - Ensure you've activated the virtual environment
   - Run `pip install -r requirements.txt` again

3. **Data file not found**
   - Run `python src/data/download_data.py` to download the dataset

## Next Steps

- Read the full documentation in README.md
- Explore the technical details in TECHNICAL_DOCS.md
- Check the Jupyter notebooks in the `notebooks/` directory for exploratory analysis 