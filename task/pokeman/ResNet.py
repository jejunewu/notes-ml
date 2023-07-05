import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model

class BasicBlock(layers.Layer):
    # 残差模块类
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        # f(x) 包含了 2 个普通卷积层，创建卷积层1
        self.conv1 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=stride, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        # 创建卷积层2
        self.conv2 = layers.Conv2D(filter_num, kernel_size=(3,3), strides=1, padding='same')
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

class ResNet(Model):
    # 通用的 ResNet 实现类
    def __init__(self, layers_dims, num_classe=5):
        super(ResNet, self).__init__()
        # 根网络, 预处理
        self.stem = Sequential([
            layers.Conv2D(64, (3, 3), strides=(1, 1)),
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')
        ])

        # 堆叠4个 ResNetBlock, 设置步长不一样
        self.layer1 = self.build_resblock(64, layers_dims[0])
        self.layer2 = self.build_resblock(128, layers_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layers_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layers_dims[3], stride=2)

        # 通过 Pooling 层将高宽降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classe)

    def call(self, inputs, training=None):
        # 前向计算函数：通过根网络
        x = self.stem(inputs)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks, stride=1):
        # 辅助函数， 堆叠 filter_num 个 BasicBlock
        res_blocks = Sequential()
        # 只有第一个BasicBlock 的步长可能不为1，实现下采样
        res_blocks.add(BasicBlock(filter_num, stride))

        # BasicBlock步长为 1
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1))
        return res_blocks


def resnet18():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([2, 2, 2, 2])

def resnet34():
    # 通过调整模块内部BasicBlock的数量和配置实现不同的ResNet
    return ResNet([3, 4, 6, 3])


