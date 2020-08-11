import tensorflow as tf

x = tf.random.normal([2, 4])
w = tf.random.normal([4, 3])

b = tf.constant([10., 20., 30.])

with tf.GradientTape() as tape:
    tape.watch([w, b])
    prob = tf.nn.softmax(x@w+b, axis=1)
    loss = tf.reduce_mean()

y = x@w + b
print(y)