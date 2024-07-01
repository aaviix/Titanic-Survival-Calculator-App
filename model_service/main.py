from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pickle import load
import numpy as np
import uvicorn

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

with open('models/Decision Tree.pkl', 'rb') as file:
    decision_tree = load(file)

with open('models/Gaussian Naive Bayes.pkl', 'rb') as file:
    gaussian_naive_bayes = load(file)

with open('models/K-Nearest Neighbors.pkl', 'rb') as file:
    k_nearest_neighbors = load(file)

with open('models/Logistic Regression.pkl', 'rb') as file:
    logistic_regression = load(file)

with open('models/Perceptron.pkl', 'rb') as file:
    perceptron = load(file)

with open('models/Random Forest.pkl', 'rb') as file:
    random_forest = load(file)

with open('models/Stochastic Gradient Descent.pkl', 'rb') as file:
    stochastic_gradient_descent = load(file)

with open('models/Support Vector Machine.pkl', 'rb') as file:
    support_vector_machine = load(file)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Passenger(BaseModel):
    pclass: int
    sex: int
    age: float
    fare: float
    traveled_alone: int
    embarked: int

def predict_survival(model_name, data):
    if model_name == "decision_tree":
        model = decision_tree
    elif model_name == "gaussian_naive_bayes":
        model = gaussian_naive_bayes
    elif model_name == "k_nearest_neighbors":
        model = k_nearest_neighbors
    elif model_name == "logistic_regression":
        model = logistic_regression
    elif model_name == "perceptron":
        model = perceptron
    elif model_name == "random_forest":
        model = random_forest
    elif model_name == "stochastic_gradient_descent":
        model = stochastic_gradient_descent
    elif model_name == "support_vector_machine":
        model = support_vector_machine
    else:
        return {"error": "Model not found"}

    X = np.array([[data.pclass, data.sex, data.age, data.fare, data.traveled_alone, data.embarked]])
    prediction = model.predict(X)
    return {"survived": bool(prediction[0])}

@app.post("/surv/{model_name}")
async def surv_or_not(model_name: str, passenger: Passenger):
    return predict_survival(model_name, passenger)

@app.get("/")
def home():
    return {"message": "Hello World"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
