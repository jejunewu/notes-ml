import tensorflow as tf
from tensorflow import keras

# x = tf.random.normal([1,5,5,3]) # 模拟输入，3通道，高宽为5
# w = tf.random.normal([3,3,3,4]) # 4个3x3大小的卷积核
# # 步长为,padding设置为输出、输入同大小
# # 需要注意的是, padding=same只有在strides=1时才是同大小
# out = tf.nn.conv2d(x,w,strides=1,padding='SAME')
# print(out)

X = tf.cast([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype='float32')
K = tf.cast([[0, 1],[2, 3]], dtype='float32')

t = tf.nn.conv1d(X,K,stride=1, padding='SAME')
print(t)

# keras.layers.Conv2D(X,K)