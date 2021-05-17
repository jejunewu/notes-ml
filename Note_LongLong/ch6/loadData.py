import tensorflow as tf
from tensorflow import keras

(x, y),(x_test, y_test) = keras.datasets.cifar10.load_data()#mnist.load_data()


# y = tf.one_hot(y, depth=10)
# print(y)
# print(x.shape, y.shape)
# print(x_test.shape, y_test.shape)


db = tf.data.Dataset.from_tensor_slices((x, y))
t = next(iter(db))[0].shape
# print(t)


(x, y),(x_test, y_test) = keras.datasets.fashion_mnist.load_data()
y = tf.one_hot(y, depth=10)
y_test = tf.one_hot(y_test, depth=10)
ds = tf.data.Dataset.from_tensor_slices((x, y))
ds = ds.shuffle(60000).batch(100)
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
ds_test = ds_test.shuffle(10000).batch(100)

print(ds, ds_test)
