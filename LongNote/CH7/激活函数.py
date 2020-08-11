
import tensorflow as tf
from matplotlib import pyplot as plt

x = tf.linspace(-10., 10., 41)
y = tf.nn.softmax(x)
y1 = tf.tanh(x)
y2 = tf.nn.relu(x)
y3 = tf.nn.leaky_relu(x)


#
print(x)
print(y2)
y = []
plt.plot(x, y)
plt.show()