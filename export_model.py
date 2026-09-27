import os
import mlflow
from mlflow import MlflowClient
import json


def main():
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mv = MlflowClient().get_model_version_by_alias("ai4i-failure", "champion")
    model_version = mv.version
    mlflow.artifacts.download_artifacts(f"models:/ai4i-failure/{mv.version}", dst_path="model")

    metadata = {
        "model_name": "ai4i-failure",
        "model_version": model_version,
        "threshold": float(os.environ.get("THRESHOLD", mv.tags.get("recommended_threshold", 0.5)))
    }
    with open("model/metadata.json", "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    main()