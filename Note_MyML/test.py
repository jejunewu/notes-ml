import numpy as np

x = np.random.random((97,2))
y = np.ones((97,1))
w = np.array([1, 1]).reshape([1,2])
print(x[:3,:])
print(y.shape)
print(w.shape)

y_new = np.dot(x, w.T)
print(y_new[:3,:])