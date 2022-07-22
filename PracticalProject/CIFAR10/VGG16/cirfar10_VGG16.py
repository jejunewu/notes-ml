import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import datasets, optimizers
from PracticalProject.CIFAR10.VGG16.VGG16 import VGG16

def prepare_cifar(x, y):
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.int32)
    return x, y

def normalize(X_train, X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    X_train = X_train / 255.
    X_test = X_test / 255.

    mean = np.mean(X_train, axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    print('mean:', mean, 'std:', std)
    X_train = (X_train - mean) / (std + 1e-7)
    X_test = (X_test - mean) / (std + 1e-7)
    return X_train, X_test

def preprocess(x, y):
    x = 2*tf.cast(x, dtype=tf.float32)/255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y


# 加载数据
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
x_train, x_test = normalize(x_train, x_test)
y_train = tf.squeeze(y_train,axis=1)
y_test  = tf.squeeze(y_test, axis=1)
print(x_train.shape,'===>', y_train.shape)
print(x_test.shape,'===>', y_test.shape)

db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).map(prepare_cifar).shuffle(50000).batch(256)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(prepare_cifar).shuffle(10000).batch(256)



model = VGG16(input_shape=(32,32,3))
model.build(input_shape=(None, 32, 32, 3))
model.summary()  # 统计网络参数

# must specify from_logits=True!
criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
metric = keras.metrics.CategoricalAccuracy()

optimizer = optimizers.Adam(learning_rate=0.0001)

for epoch in range(50):

    for step, (x, y) in enumerate(db_train):

        y = tf.one_hot(y, depth=10)
        with tf.GradientTape() as tape:
            logits = model(x)
            loss = criteon(y, logits)

            # loss = tf.reduce_mean(loss)
            # loss2 = compute_loss(logits, tf.argmax(y, axis=1))
            # mse_loss = tf.reduce_sum(tf.square(y-logits))
            # print(y.shape, logits.shape)
            metric.update_state(y, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        # MUST clip gradient here or it will disconverge!
        # grads = [tf.clip_by_norm(g, 15) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 40 == 0:
            # for g in grads:
            #     print(tf.norm(g).numpy())
            print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
            metric.reset_states()

    if epoch % 1 == 0:

        metric = keras.metrics.CategoricalAccuracy()
        for x, y in db_test:
            # [b, 10]
            y = tf.one_hot(y, depth=10)

            logits = model.predict(x)
            # be careful, these functions can accept y as [b] without warnning.
            metric.update_state(y, logits)
        print('test acc:', metric.result().numpy())
        metric.reset_states()

