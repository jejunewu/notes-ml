import tensorflow as tf
from tensorflow.keras import layers, datasets, Sequential, losses
import matplotlib.pyplot as plt

# 加载数据
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
print(x_train.shape,'===>', y_train.shape)
print(x_test.shape,'===>', y_test.shape)

# 归一化
x_train = x_train/255.
x_test = x_test/255.
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(50).shuffle(32)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)


# 定义网络
netWork = Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
])

netWork.summary()
netWork.compile(optimizer='SGD', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
netWork.fit(db_train, batch_size=50,epochs=10, validation_data=db_test)