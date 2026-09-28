from datetime import datetime
import json
import os
from datetime import timezone
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, Request
import mlflow
from mlflow import MlflowClient
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
import logging
import pandas as pd

FEATURE_MAP = {"air_temp": "Air temperature [K]", "process_temp": "Process temperature [K]",
               "rot_speed": "Rotational speed [rpm]", "torque": "Torque [Nm]", "tool_wear": "Tool wear [min]",
               "product_type": "Type"}

class MachineReading(BaseModel):
    air_temp: float= Field(gt=0)
    process_temp: float = Field(gt=0)
    rot_speed: float = Field(gt=0)
    torque: float = Field(gt=0)
    tool_wear: float = Field(ge=0)
    product_type: Literal["L", "M", "H"]

class PredictionResponse(BaseModel):
    failure_probability: float
    failure_predicted: bool
    threshold: float
    model_version: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    prediction_log_path = os.environ.get("PREDICTION_LOG_PATH", Path(__file__).resolve().parent.parent / "logs/prediction.jsonl")
    os.makedirs(os.path.dirname(prediction_log_path), exist_ok=True)
    logger = logging.getLogger("prediction_log")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    fh = logging.FileHandler(prediction_log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    app.state.prediction_logger = logger
    try:
        model_name = os.environ.get("MODEL_NAME", "ai4i-failure")
        model_alias = os.environ.get("MODEL_ALIAS", "champion")
        if "MODEL_PATH" not in os.environ:
            source_uri = f"{model_name}@{model_alias}"
            mv = MlflowClient().get_model_version_by_alias(model_name, model_alias)
            app.state.model_version = mv.version
            app.state.model = mlflow.sklearn.load_model(f"models:/{model_name}/{mv.version}")
            app.state.threshold = float(os.environ.get("THRESHOLD", mv.tags.get("recommended_threshold", 0.5)))
        else:
            source_uri = os.environ["MODEL_PATH"]
            app.state.model = mlflow.sklearn.load_model(os.environ["MODEL_PATH"])
            with open(f'{os.environ["MODEL_PATH"]}/metadata.json', "r") as f:
                metadata = json.load(f)
            app.state.model_version = metadata['model_version']
            app.state.threshold = metadata['threshold']
    except Exception as e:
        raise RuntimeError(
            f"Could not load model {source_uri}"
            f"from tracking URI '{mlflow.get_tracking_uri()}'") from e
    yield

    logger .removeHandler(fh)
    fh.close()
app = FastAPI(lifespan=lifespan)

def map_data(machine_reading: MachineReading):
    row = {FEATURE_MAP[k]: v for k, v in machine_reading.model_dump().items()}
    return pd.DataFrame([row])

def log_prediction(reading: MachineReading, request: Request, proba: float):
   request.app.state.prediction_logger.info(json.dumps({
       "timestamp": datetime.now(timezone.utc).isoformat(),
       "model_version": request.app.state.model_version,
       "threshold": request.app.state.threshold,
       "input": reading.model_dump(),
       "failure_probability": proba,
       "failure_predicted": proba >= request.app.state.threshold

   })
   )

@app.get("/")
def hello_world():
    return {"message": "Hello World"}

@app.post("/predict", response_model=PredictionResponse)
def predict(reading: MachineReading, request: Request):
    state = request.app.state
    X = map_data(reading)

    proba = float(state.model.predict_proba(X)[0, 1])

    log_prediction(reading, request, proba)

    return PredictionResponse(
        failure_probability=proba,
        failure_predicted=proba >= state.threshold,
        threshold=state.threshold,
        model_version=str(state.model_version)
    )

@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_version": request.app.state.model_version}