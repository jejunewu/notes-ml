import tensorflow as tf
from tensorflow.keras import layers, Sequential

class BasicBlock(layers.Layer):
    # 残差模块类
    def __init__(self, filter_num, stride):
        super(BasicBlock, self).__init__()

        # f(x) 包含了 2 个普通卷积层，创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 创建卷积层2
        self.conv2 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, padding='same')
        self.bn2 = layers.BatchNormalization()

        # 插入identity(x)层
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride))
        else:
            self.downsample = lambda x:x

    def call(self, inputs, traning=None):
        # 前向传播函数
        # 第一个卷积层
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        # 第二个卷积层
        out = self.conv2(out)
        out = self.bn2(out)
        # identity(x)转换
        identity = self.downsample(inputs)
        # f(x) + x 运算
        output = layers.add([out, identity])
        # 通过激活函数并返回
        output = tf.nn.relu(output)
        return output
