import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets,layers,optimizers, models, regularizers, Sequential



class VGG16(models.Model):
    def __init__(self, input_shape):

        super(VGG16, self).__init__()
        weight_decay = 0.000
        self.num_classes = 10

        model = Sequential()
        # Conv-Conv-Pooling 单元1
        # 64 个3x3 卷积核, 输入输出同大小
        # 高宽减半
        model.add(layers.Conv2D(64, kernel_size=[3, 3], padding='same',input_shape=input_shape, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.3))
        model.add(layers.Conv2D(64, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=[2, 2]))


        # Conv-Conv-Pooling 单元2,输出通道提升至128，
        # 高宽大小减半
        model.add(layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Conv2D(128, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.MaxPool2D(pool_size=[2, 2]))

        # Conv-Conv-Pooling 单元3,输出通道提升至256，
        # 高宽大小减半
        model.add(layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Conv2D(256, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPool2D(pool_size=[2, 2]))

        # Conv-Conv-Pooling 单元4,输出通道提升至512，
        # 高宽大小减半
        model.add(layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPool2D(pool_size=[2, 2]))

        # Conv-Conv-Pooling 单元5,输出通道提升至512，
        # 高宽大小减半
        model.add(layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.Conv2D(512, kernel_size=[3, 3], padding='same', activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.4))
        model.add(layers.MaxPool2D(pool_size=[2, 2]))
        model.add(layers.Dropout(0.5))

        # 创建3 层全连接层子网络
        model.add(layers.Flatten())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))

        model.add(layers.Dense(self.num_classes))

        self.model = model

    def call(self, x):
        x = self.model(x)
        return x