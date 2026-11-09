import pandas as pd
import numpy as np
import uuid
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo


print("Training Diabetes Model...")
diabetes = pd.read_csv("data/pima-indians-diabetes.data.csv",
                       names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
                              "DiabetesPedigreeFunction", "Age", "Outcome"])
X_d, y_d = diabetes.drop("Outcome", axis=1), diabetes["Outcome"]

print(list(X_d))
print(X_d)
print(y_d)
print(f"{sum(y_d)}/{len(y_d)} = {sum(y_d)/len(y_d)}")

# Breast Cancer Model
print("Training Breast Cancer Model...")
breast = load_breast_cancer()
X_b = pd.DataFrame(breast.data, columns=breast.feature_names)
y_b = breast.target  # 0 = malignant, 1 = benign

print(list(X_b))
print(X_b)
print(y_b)
print(f"{sum(y_b)}/{len(y_b)} = {sum(y_b)/len(y_b)}")

# Heart Model
print("Training Heart Model...")
heart = pd.read_csv("data/heart.csv")
X_h, y_h = heart.drop("target", axis=1), heart["target"]

print(list(X_h))
print(X_h)
print(y_h)
print(f"{sum(y_h)}/{len(y_h)} = {sum(y_h)/len(y_h)}")

# Parkinson Model
print("Training Parkinson Model...")
parkinsons = fetch_ucirepo(id=174) 
  
# data (as pandas dataframes) 
X_p = parkinsons.data.features 
y_p = parkinsons.data.targets["status"]


print(list(X_p))
print(X_p)
print(y_p)
print(f"{sum(y_p)}/{len(y_p)} = {sum(y_p)/len(y_p)}")
exit()