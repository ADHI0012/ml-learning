import numpy as np

def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0
    
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost += (f_wb_i - y[i]) ** 2
        
    total_cost = cost / (2 * m)
    return total_cost

def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n, ))
    dj_db = 0
    
    for i in range(n): 
        for j in range(m):
            error = (np.dot(X[j], w) + b) - y[j]
            if i == 0:
                dj_db += error
            dj_dw[i] += error * X[j,i]
            
    dj_dw /= m
    dj_db /= m
        
    return dj_dw, dj_db
            
            
            
    
    dj_dw /= m
    dj_db /= m
    
    return dj_dw, dj_db


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(X, y, w, b)
        
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        
    return w, b
        
        
        
    


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])
iterations = 1000
initial_w, initial_b = np.array([0,0,0,0]), 0.
alpha = 5.0e-7


final_w, final_b = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)

for w in final_w:
    print(f"{w: 0.2f}", end=" ")
print()
print(f"b = {final_b: 0.2f}")