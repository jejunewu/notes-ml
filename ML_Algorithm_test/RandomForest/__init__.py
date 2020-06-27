import numpy as np

x = np.ones([12,3])
y = np.zeros([5,3])
z = np.ones([4,3])*2
t = np.concatenate((x,y,z))
print(t)