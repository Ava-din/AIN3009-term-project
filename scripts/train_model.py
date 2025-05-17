import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from preprocess import load_and_preprocess_data


def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train the model and log to MLflow.
    """
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        roc = roc_auc_score(y_test, preds)

        # Log parameters and metrics
        mlflow.log_param("model_name", model_name)
        if hasattr(model, "get_params"):
            for param, value in model.get_params().items():
                mlflow.log_param(param, value)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc)

        # Log the model artifact
        mlflow.sklearn.log_model(model, "model")

        print(f"Model {model_name} logged to MLflow.")


def main():
    # Set experiment name (creates if not exists)
    mlflow.set_experiment("Diabetes_Prediction")

    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Define models to train
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    # Train and log each model
    for model_name, model in models.items():
        train_and_log_model(model, model_name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
