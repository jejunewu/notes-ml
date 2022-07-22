import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets, optimizers, metrics, Sequential

### 导入数据
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(x_train.shape, '===>', y_train.shape)
print(x_test.shape, '===>', y_test.shape)

### 特征处理
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255
x_test =  tf.convert_to_tensor(x_test, dtype=tf.float32) / 255

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(32).batch(50)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).shuffle(32).batch(50)


class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()

        self.model = keras.Sequential([
            layers.Dense(100, activation='relu'),
            layers.Dense(100, activation='relu'),
            layers.Dense(10, activation='softmax')])

    def call(self, x, training=None):
        x = tf.reshape(x, [-1, 28*28])

        x = self.model(x, training)

        return x

#from Example.MNIST.network import MyModel
network = MyModel()
network.build(input_shape=(28, 28))
network.summary()



### 搭建网络 - keras版
network.compile(optimizer='SGD', loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
network.fit(db_train, epochs=10, batch_size=100, validation_data=db_test)

print('###############################################################################')

### 搭建网络-手动版
optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()
for step, (x, y) in enumerate(db_train):
    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 28*28]
        x = tf.reshape(x, (-1, 28*28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # [b, 10]
        loss = tf.square(out - y_onehot)
        # [b]
        loss = tf.reduce_sum(loss)

    acc_meter.update_state(tf.argmax(out, axis=1), y)
    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))

    if step % 200 == 0:
        print(step, 'loss: ', float(loss), 'acc: ',acc_meter.result().numpy())
        acc_meter.reset_states()