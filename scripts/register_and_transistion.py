import mlflow
from mlflow.tracking import MlflowClient
import time


def register_best_model(experiment_name, model_name="Diabetes_RF_Model"):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist")
    experiment_id = experiment.experiment_id

    # Find best run by accuracy (descending)
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["metrics.accuracy DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No runs found for experiment '{experiment_name}'")

    best_run = runs[0]
    run_id = best_run.info.run_id
    best_acc = best_run.data.metrics["accuracy"]
    print(f"Best run_id: {run_id} with accuracy: {best_acc}")

    model_uri = f"runs:/{run_id}/model"

    # Register model (create model if not exist)
    try:
        client.create_registered_model(model_name)
        print(f"Registered new model: {model_name}")
    except mlflow.exceptions.RestException:
        print(f"Model '{model_name}' already registered")

    # Create a new model version
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id,
    )
    print(f"Created model version: {mv.version}")

    return mv.version, client, model_name


def transition_model_version_stage(client, model_name, version, stage):
    print(f"Transitioning model {model_name} version {version} to stage: {stage}...")
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=True,  # archive old versions in that stage
    )
    print(f"Model {model_name} version {version} transitioned to {stage}")


def main():
    experiment_name = "Diabetes_Prediction_Hyperopt"
    model_name = "Diabetes_RF_Model"

    version, client, model_name = register_best_model(experiment_name, model_name)

    # Transition to Staging
    transition_model_version_stage(client, model_name, version, "Staging")

    # Optional: Wait or perform validation here
    print("Waiting 5 seconds before promoting to Production...")
    time.sleep(5)

    # Transition to Production
    transition_model_version_stage(client, model_name, version, "Production")


if __name__ == "__main__":
    main()
