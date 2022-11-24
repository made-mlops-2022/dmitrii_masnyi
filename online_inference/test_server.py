from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from main import app

client = TestClient(app)

data = [[69,1,0,160,234,1,2,131,0,0.1,1,1,0,0],
        [66,0,0,150,226,0,0,114,0,2.6,2,0,0,0],
        [61,1,0,134,234,0,0,145,0,2.6,1,2,0,1]]
labels = ["age", "sex", "cp", "trestbps", "chol", "fbs", 
          "restecg", "thalach", "exang", "oldpeak", 
          "slope", "ca", "thal", "condition"]

def test_read_main():
    with client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == "It is entry point of our predictor"

def test_health_state():
    with client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == 1

def test_valid_request():
    global data
    global labels
    with client:
        data = pd.DataFrame(data, columns=labels)
        data = data.iloc[:, :-1]
        req = {"data": [data.iloc[0].to_list()], "features": list(data.columns)}
        response = client.post("/predict/", json=req)
        assert response.status_code == 200
        assert response.json() == [{'disease': 1}]

        req = {"data": [data.iloc[1].to_list()], "features": list(data.columns)}
        response = client.post("/predict/", json=req)
        assert response.status_code == 200
        assert response.json() == [{'disease': 0}]

        req = {"data": [data.iloc[2].to_list()], "features": list(data.columns)}
        response = client.post("/predict/", json=req)
        assert response.status_code == 200
        assert response.json() == [{'disease': 1}]

def test_invalid_rerquest():
    with client:
        req = {"data": [], "features": 1}
        response = client.post("/predict/", json=req)
        assert response.status_code == 422

        req = {}
        response = client.post("/predict/", json=req)
        assert response.status_code == 422



