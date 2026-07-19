import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_cost(X,y,w,b):
    m = X.shape[0]
    cost = 0.0
    
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1-y[i]) * np.log(1-f_wb_i)
        
    cost /= m   
    return cost

def compute_gradient(X,y,w,b):
    m,n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0.0
    
    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        error = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def gradient_descent(X, y, w_in, b_in, num_iters, alpha):
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    
    return w, b


X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  
y_train = np.array([0, 0, 0, 1, 1, 1])
w_temp  = np.zeros_like(X_train[0])
b_temp = 0.
alpha = 0.1
iters = 10000

w,b = gradient_descent(X_train, y_train, w_temp, b_temp, iters, alpha)

print(f"The value of W: {w}")
print(f"The value of b: {b}")

print(compute_cost(X_train, y_train, w, b))