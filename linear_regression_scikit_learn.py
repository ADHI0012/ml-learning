import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


dataset = pd.read_csv("data/housing_data.csv")

X_train = dataset.iloc[:, :3]
y_train = dataset.iloc[:, 3]

scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

sgdr = SGDRegressor(max_iter=10000)
sgdr.fit(X_norm, y_train)

b = sgdr.intercept_
w = sgdr.coef_

print(f"Intercept: {b}")
print(f"W values: {w}")

y_pred = sgdr.predict(X_norm)
print("Actual V Predicted")
for i in range(len(y_pred)):
    print(f"Actual: {y_train.iloc[i]} | Predicted: {y_pred[i]:.2f}")





