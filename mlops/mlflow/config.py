import mlflow

# Local MLflow tracking
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Experiment name
mlflow.set_experiment("Politician-Image-Classification")