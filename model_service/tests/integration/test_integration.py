import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"  # Ensure your app is running on this URL

payload = {
    "pclass": 3,
    "sex": 0,
    "age": 22.0,
    "fare": 7.25,
    "traveled_alone": 1,
    "embarked": 2
}
def test_integration_predict_logistic_regression():
    response = requests.post(f"{BASE_URL}/surv/logistic_regression", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_integration_predict_decision_tree():
    response = requests.post(f"{BASE_URL}/surv/decision_tree", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_integration_predict_gaussian_naive_bayes():
    response = requests.post(f"{BASE_URL}/surv/gaussian_naive_bayes", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_integration_predict_k_nearest_neighbors():
    response = requests.post(f"{BASE_URL}/surv/k_nearest_neighbors", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_integration_predict_perceptron():
    response = requests.post(f"{BASE_URL}/surv/perceptron", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_integration_predict_random_forest():
    response = requests.post(f"{BASE_URL}/surv/random_forest", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_integration_predict_stochastic_gradient_descent():
    response = requests.post(f"{BASE_URL}/surv/stochastic_gradient_descent", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]

def test_integration_predict_support_vector_machine():
    response = requests.post(f"{BASE_URL}/surv/support_vector_machine", json=payload)
    assert response.status_code == 200
    assert "survived" in response.json()
    assert response.json()["survived"] in [True, False]
