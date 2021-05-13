import  tensorflow as tf
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

(train_x, train_y), (test_x, test_y) = datasets.mnist.load_data()
print(train_x.shape, '--->', train_y.shape)
print(test_x.shape, '--->', test_y.shape)

# reshape
train_x = tf.reshape(train_x, (60000, 28, 28, 1))
test_x = tf.reshape(test_x, (10000, 28, 28, 1))
print('reshape 之后：')
print(train_x.shape, '--->', train_y.shape)
print(test_x.shape, '--->', test_y.shape)

# print(train_x[0])
# C1
conv_layers=[
    #C1
    layers.Conv2D(6, kernel_size=[5, 5], padding="same", activation=tf.nn.tanh),
    #S2
    layers.AveragePooling2D(pool_size=[2, 2], strides=2),
    #C3
    layers.Conv2D(16, kernel_size=[5, 5], padding="same", activation=tf.nn.tanh),
    #S4
    layers.AveragePooling2D(pool_size=[2, 2], strides=2),
    #C5
    layers.Conv2D(32, kernel_size=[5, 5], padding="same", activation=tf.nn.tanh),
    #F6
    layers.Flatten(),
    layers.Dense(120, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
]


conv_net = Sequential(conv_layers)
conv_net.build(input_shape=[None, 28, 28, 1])
conv_net.summary()
conv_net.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
conv_net.fit(train_x,train_y,epochs=5,validation_split=0.1)
# testLoss, testAcc = conv_net.evaluate(test_x, test_y)
# print(testAcc)