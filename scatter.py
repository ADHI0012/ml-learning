import matplotlib.pyplot as plt
import numpy as np


x = np.arange(0, 20, 1)
y = x**2

X = np.c_[x, x**2, x**3]
print(X)

fig,ax = plt.subplots(1,3,figsize=(12, 3), sharey=True)

for i in range(len(ax)):
    ax[i].scatter(X[:, i], y)
    
plt.show()