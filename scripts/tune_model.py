import mlflow
import mlflow.sklearn
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from preprocess import load_and_preprocess_data

# Load data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Define search space
search_space = {
    "n_estimators": hp.choice("n_estimators", [50, 100, 150, 200]),
    "max_depth": hp.choice("max_depth", [5, 10, 15, 20, None]),
    "max_features": hp.choice("max_features", ["sqrt", "log2", None]),
}


def objective(params):
    with mlflow.start_run(nested=True):  # Nested run inside global run
        clf = RandomForestClassifier(**params, random_state=42)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log hyperparameters and metric
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # Log the model
        mlflow.sklearn.log_model(clf, "model")

        return {"loss": -acc, "status": STATUS_OK}


def main():
    mlflow.set_experiment("Diabetes_Prediction_Hyperopt")

    with mlflow.start_run(run_name="hyperopt_random_forest"):
        trials = Trials()
        best = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=20,
            trials=trials,
        )

        print("Best hyperparameters:", best)


if __name__ == "__main__":
    main()
