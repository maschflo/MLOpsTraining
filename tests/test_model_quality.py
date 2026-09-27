import json
import mlflow
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def test_model_quality():
    with open(f'{MODEL_DIR}/metadata.json', "r") as f:
        metadata = json.load(f)
    data = pd.read_csv(DATA_DIR / "production_data.csv")
    model = mlflow.sklearn.load_model(MODEL_DIR)
    proba = model.predict_proba(data)[:, 1]
    pred = (proba >= metadata["threshold"]).astype(int)
    y = data['Machine failure']
    assert f1_score(y, pred) >= 0.7
    assert precision_score(y, pred) >= 0.7
    assert recall_score(y, pred) >= 0.7
