# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# %matplotlib inline
from pickle import dump

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import os

# Directories
model_dir = 'models'
predictions_dir = 'test predictions'

# Create directories if they don't exist
os.makedirs(model_dir, exist_ok=True)
os.makedirs(predictions_dir, exist_ok=True)

# Load datasets
train_df = pd.read_csv('SWE dataset/train.csv')
test_df = pd.read_csv('SWE dataset/test.csv')

# Data preprocessing function
def preprocess_data(df):
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['Traveled Alone'] = np.where((df['SibSp'] + df['Parch']) > 0, 0, 1)
    return df

# Preprocess both train and test data
train_df = preprocess_data(train_df)
test_df = preprocess_data(test_df)

# Select features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Traveled Alone', 'Embarked']
X_train = train_df[features]
y_train = train_df['Survived']
X_test = test_df[features]

# Train multiple models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Gaussian Naive Bayes": GaussianNB(),
    "Perceptron": Perceptron(),
    "Stochastic Gradient Descent": SGDClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

# Train models and make predictions
predictions = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    with open(f'{model_dir}/{model_name}.pkl', 'wb') as file:
        dump(model, file)
    predictions[model_name] = model.predict(X_test)
    print(f"{model_name} predictions: {predictions[model_name][:10]}")  # Display first 10 predictions for each model

# Save predictions to CSV
for model_name, prediction in predictions.items():
    submission_df = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': prediction
    })
    submission_df.to_csv(f'{predictions_dir}/{model_name}_predictions.csv', index=False)

# Display a few predictions from each model
for model_name, prediction in predictions.items():
    print(f"\n{model_name} predictions:\n", pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': prediction
    }).head())
