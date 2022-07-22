from matplotlib import pyplot as plt
import numpy as np

x = np.linspace(-5,5,100)
y = np.sin(x)+np.random.rand(100)*0.1

plt.figure()
plt.plot(x,y,'x')
plt.show()


