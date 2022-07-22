import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential,Model


class MyModel(keras.Model):  # 继承自Model
    def __init__(self):
        super(MyModel, self).__init__()

        # 使用自定义全连接层
        self.fc1 = layers.Flatten()
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(256, activation='relu')
        self.fc4 = layers.Dense(10, activation='softmax')

    def call(self, inputs, training=None):
        # 网络的叠加
        x = self.fc1(input_shape=inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
