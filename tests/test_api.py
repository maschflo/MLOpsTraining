from pathlib import Path
import mlflow
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from api.main import app

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


@pytest.fixture(scope="module")
def client():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODEL_PATH", str(MODEL_DIR))
        mp.delenv("THRESHOLD", raising=False)
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_predict(client):
    response = client.post("/predict", json={
        "air_temp": 298.7,
        "process_temp": 309.8,
        "rot_speed": 1354,
        "torque": 53.3,
        "tool_wear": 212,
        "product_type": "L"
    })
    body = response.json()
    assert response.status_code == 200
    assert 0 <= body.get("failure_probability") <= 1


def test_predict_fail(client):
    fail_response = client.post("/predict", json={
        "air_temp": 298.7,
        "process_temp": 309.8,
        "rot_speed": 1354,
        "torque": 53.3,
        "tool_wear": 212,
        "product_type": "X"
    })
    assert fail_response.status_code == 422


def test_skew(client):
    response = client.post("/predict", json={
        "air_temp": 298.7,
        "process_temp": 309.8,
        "rot_speed": 1354,
        "torque": 53.3,
        "tool_wear": 212,
        "product_type": "L"
    })
    body = response.json()
    model = mlflow.sklearn.load_model(MODEL_DIR)
    row = {"Air temperature [K]": 298.7, "Process temperature [K]": 309.8, "Rotational speed [rpm]": 1354,
                   "Torque [Nm]": 53.3, "Tool wear [min]": 212, "Type": "L"}
    X = pd.DataFrame([row])
    proba = float(model.predict_proba(X)[0, 1])
    assert body.get("failure_probability") == pytest.approx(proba)
