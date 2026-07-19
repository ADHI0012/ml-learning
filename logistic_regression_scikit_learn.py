import numpy as np
from sklearn.linear_model import LogisticRegression

X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  
y_train = np.array([0, 0, 0, 1, 1, 1])

model = LogisticRegression()
model.fit(X_train, y_train)


print(f"W value: {model.coef_}")
print(f"b value: {model.intercept_}")

y_pred = model.predict(X_train)
print(f"Prediction on training set: {y_pred}")



# The W value from scikit-learn and gradient descent implementation differ because the dataset
# is a linearly seperable data, and there are infinitely many lines that can seperate the positive and negative outcome