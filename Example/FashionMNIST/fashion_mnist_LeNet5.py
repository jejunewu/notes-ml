import tensorflow as tf
from tensorflow.keras import layers, datasets, Sequential, losses
import matplotlib.pyplot as plt

# 加载数据
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
# reshape
print(x_train.shape,'===>', y_train.shape)
print(x_test.shape,'===>', y_test.shape)
x_train = tf.reshape(x_train, (60000, 28, 28, 1))
x_test = tf.reshape(x_test, (10000, 28, 28, 1))

# 归一化
x_train = x_train/255
x_test = x_test/255
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(50).shuffle(32)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)


# 定义网络
netWork = Sequential([
    # C1   28*28 => 6@28*28
    layers.Conv2D(filters=6, kernel_size=[5,5], padding='same'),
    # S2   6@28*28 => 6@14*14
    layers.MaxPool2D(pool_size=[2,2], strides=2),
    # C3   6@14*14 => 16@10*10
    layers.Conv2D(filters=16, kernel_size=[5,5]),
    # S4   16@10*10 => 16@5*5
    layers.MaxPool2D(pool_size=[2, 2], strides=2),
    # C5   16@5*5 => 120
    layers.Conv2D(filters=16, kernel_size=[5,5], padding='same'),
    layers.Flatten(),
    layers.Dense(120),
    # F6
    layers.Dense(84),
    layers.Dense(10, activation='softmax')
])

netWork.build(input_shape=[None, 28, 28, 1])
netWork.summary()
netWork.compile(optimizer='SGD', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
netWork.fit(db_train, batch_size=50,epochs=10, validation_data=db_test)