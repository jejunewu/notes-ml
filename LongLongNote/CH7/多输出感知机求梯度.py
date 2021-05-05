import tensorflow as tf

x = tf.random.normal([2,4])
w = tf.random.normal([4,3])
b = tf.zeros([3])
y = tf.constant([1, 0])
print(b)

with tf.GradientTape() as tape:
    tape.watch([w,b])
    prob = tf.nn.softmax(x@w+b,axis=1)
    loss = tf.reduce_mean(tf.losses.MSE(tf.one_hot(y, depth=3), prob))

grads = tape.gradient(loss, [w,b])

print(grads)