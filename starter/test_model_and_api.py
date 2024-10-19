import pytest
from fastapi.testclient import TestClient
from main import app
from starter.ml.model import train_model, compute_model_metrics, inference
import numpy as np
import pandas as pd
from starter.ml.data import process_data

client = TestClient(app)

def test_train_model():
    # Create dummy data
    X = np.random.rand(100, 10)
    y = np.random.randint(2, size=100)
    
    model = train_model(X, y)
    assert model is not None

def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    
    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_inference():
    # Create dummy data and model
    X = np.random.rand(100, 10)
    y = np.random.randint(2, size=100)
    model = train_model(X, y)
    
    # Test inference
    X_test = np.random.rand(10, 10)
    predictions = inference(model, X_test)
    assert len(predictions) == 10
    assert all(pred in [0, 1] for pred in predictions)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Census Income Prediction API"}

def test_predict_low_income():
    response = client.post("/predict", json={
        "age": 25,
        "workclass": "Private",
        "fnlwgt": 226802,
        "education": "11th",
        "education-num": 7,
        "marital-status": "Never-married",
        "occupation": "Machine-op-inspct",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    })
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]

def test_predict_high_income():
    response = client.post("/predict", json={
        "age": 45,
        "workclass": "Private",
        "fnlwgt": 160323,
        "education": "Masters",
        "education-num": 14,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 10000,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States"
    })
    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]