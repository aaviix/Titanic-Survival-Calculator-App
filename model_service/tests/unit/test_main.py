import sys
import os
import pytest

# Add the directory containing 'main.py' to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from main import app  # assuming you are using FastAPI
from fastapi.testclient import TestClient

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}

def test_predict_survival_logistic_regression():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/logistic_regression", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_predict_survival_decision_tree():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/decision_tree", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_predict_survival_gaussian_naive_bayes():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/gaussian_naive_bayes", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_predict_survival_k_nearest_neighbors():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/k_nearest_neighbors", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_predict_survival_perceptron():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/perceptron", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_predict_survival_random_forest():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/random_forest", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_predict_survival_stochastic_gradient_descent():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/stochastic_gradient_descent", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_predict_survival_support_vector_machine():
    payload = {
        "pclass": 3,
        "sex": 0,
        "age": 22.0,
        "fare": 7.25,
        "traveled_alone": 1,
        "embarked": 2
    }
    response = client.post("/surv/support_vector_machine", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]
