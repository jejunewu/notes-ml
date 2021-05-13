import  tensorflow as tf
from tensorflow import keras
from    tensorflow.keras import layers, optimizers, datasets, Sequential
import  os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(x_train.shape, '===>', y_train.shape)
print(x_test.shape, '===>', y_test.shape)

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255
x_test =  tf.convert_to_tensor(x_test, dtype=tf.float32) / 255

# reshape
x_train = tf.reshape(x_train, [60000,28,28,1])
x_test = tf.reshape(x_test, [10000,28,28,1])

db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(32).batch(100)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(100)


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
        #x = tf.reshape(x, [None, 28, 28, 1])

        x = self.model(x, training)

        return x


conv_net = MyLeNet5()
conv_net.build(input_shape=(100, 28, 28, 1))
conv_net.summary()
#conv_net.compile(optimizer='adam',loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#conv_net.fit(db_train,epochs=5,batch_size=500, validation_data=db_test)
# testLoss, testAcc = conv_net.evaluate(test_x, test_y)
# print(testAcc)