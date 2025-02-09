import pytest
import requests
import time

# Use Docker service names
MAIN_BACKEND_URL = "http://web_backend:8080"
MODEL_BACKEND_URL = "http://model_backend:8000"

# Wait for services to be ready
time.sleep(60)

def test_main_backend_to_model_backend():
    # Ensure both servers are running before this test
    passenger_data = {
        "pclass": 1,
        "sex": 0,
        "age": 30,
        "fare": 100,
        "traveled_alone": 1,
        "embarked": 2
    }
    model_name = "random_forest"

    response = requests.post(f"{MAIN_BACKEND_URL}/surv_or_not/{model_name}", json=passenger_data)
    assert response.status_code == 200, f"Failed with status code {response.status_code}, response: {response.text}"
    response_json = response.json()
    assert "survived" in response_json, f"Response JSON does not contain 'survived': {response_json}"

    model_response = requests.post(f"{MODEL_BACKEND_URL}/surv/{model_name}", json=passenger_data)
    assert model_response.status_code == 200, f"Failed with status code {model_response.status_code}, response: {model_response.text}"
    model_response_json = model_response.json()
    assert response_json["survived"] == model_response_json["survived"], (
        f"Mismatch between main backend and model backend responses: {response_json} vs {model_response_json}"
    )
