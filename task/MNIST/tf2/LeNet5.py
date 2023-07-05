import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets, optimizers, metrics, Sequential

class MyLeNet5(keras.Model):

    def __init__(self):
        super(MyLeNet5, self).__init__()

        self.model = keras.Sequential([
            # C1
            layers.Conv2D(6, kernel_size=[5, 5], padding="same", activation=tf.nn.tanh),
            # S2
            layers.AveragePooling2D(pool_size=[2, 2], strides=2),
            # C3
            layers.Conv2D(16, kernel_size=[5, 5], padding="same", activation=tf.nn.tanh),
            # S4
            layers.AveragePooling2D(pool_size=[2, 2], strides=2),
            # C5
            layers.Conv2D(32, kernel_size=[5, 5], padding="same", activation=tf.nn.tanh),
            # F6
            layers.Flatten(),
            layers.Dense(120, activation=tf.nn.relu),
            layers.Dense(10, activation=tf.nn.softmax)
        ])

    def call(self, x, training=None):
        x = tf.reshape(x, [None, 28, 28, 1])

        x = self.model(x, training)

        return x