import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
import matplotlib.pyplot as plt
import os

def show_pic(pic,name='img',cmap=None):
    '''显示图像'''
    plt.imshow(pic/255,cmap=cmap)
    plt.axis('off')   # 打开坐标轴为 on
# 设置图像标题
    plt.title('%s'%(name))
    plt.show()

### 重命名图像
# oldFileNameList = os.listdir(data_dir+'train\\bees')
# for i,e in enumerate(oldFileNameList):
#     oldFileName = data_dir + 'train\\bees\\'+e
#     newFileName = data_dir + 'train\\bees\\'+'bees_'+str(i)+'.jpg'
#     print(oldFileName,'-->' ,newFileName)
#     os.rename(oldFileName, newFileName)

### 4.1数据载入
data_dir = 'C:\\Users\\ThinkPad\\Desktop\\hymenoptera_data\\'
train_ants_dir = data_dir+'train\\ants\\'
train_bees_dir = data_dir+'train\\bees\\'
train_data = [[],[]]
for i,e in enumerate(os.listdir(train_ants_dir)):
    img = tf.io.read_file(train_ants_dir + e)
    img = tf.image.decode_jpeg(img, channels=3)
    train_data[0].append(img)
    train_data[1].append(0)
for i,e in enumerate(os.listdir(train_bees_dir)):
    img = tf.io.read_file(train_bees_dir + e)
    img = tf.image.decode_jpeg(img, channels=3)
    train_data[0].append(img)
    train_data[1].append(1)
train_data = np.array(train_data)
print(train_data.shape)


img = tf.io.read_file('ants_0.jpg')
img = tf.image.decode_jpeg(img, channels=3)
old = img.shape
img = tf.image.random_flip_left_right(img)
img = tf.image.resize(img, [512, 512])
print(old,'-->',img.shape)
show_pic(img)
ants = os.listdir(data_dir+'train\\ants\\')
# for i,e in enumerate(ants):
#     img = tf.io.read_file(data_dir + 'train\\ants\\' +e)
#     img = tf.image.decode_jpeg(img, channels=3)
#     old = img.shape
#     img = tf.image.random_flip_left_right(img)
#     img = tf.image.resize(img, [512, 512])
#     print(old, '-->', img.shape)
#     show_pic(img)
# print(ants)
# tf.io.read_file()

def preprocess(x,y):
    # x: 图片的路径，y：图片的数字编码
    x = tf.io.read_file(x)
    x = tf.image.decode_jpeg(x, channels=3) # RGBA
    x = tf.image.resize(x, [244, 244])

    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    x = tf.image.random_crop(x, [224,224,3])

    # x: [0,255]=> -1~1
    x = tf.cast(x, dtype=tf.float32) / 255.
    # x = normalize(x)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=2)

    return x, y

