# Importing Libraries
import math, copy
import numpy as np
import matplotlib.pyplot as plt


def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i])**2
    
    total_cost = cost / (2 * m)
    return total_cost

def compute_gradient(x,y,w,b):
    dj_dw = dj_db = 0
    m = x.shape[0]
    
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = (f_wb - y[i])
        dj_dw += dj_dw_i
        dj_db += dj_db_i
        
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w,b = w_in, b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x,y,w,b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
        
    return w,b


        
# Training set
x_train = np.array([1.0, 2.0]) #features
y_train = np.array([300.0, 500.0]) #target value


w,b = gradient_descent(x_train, y_train, 0, 0, 1.0e-2, 10000, compute_cost, compute_gradient)

print(f"w = {w:.2f}, b = {b:.2f}")

print("Prediction for x=1:", w*1 + b)
print("Prediction for x=2:", w*2 + b)
