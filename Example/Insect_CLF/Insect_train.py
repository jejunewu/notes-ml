import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers,metrics,Sequential,layers
from Example.Insect_CLF.Insect import load_insect
import os
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
'''
@:param  x: 图片的路径List，
@:param  y：图片的数字编码List
@:return 
'''

def preprocess(x, y):
    x = tf.io.read_file(x) # 根据路径读取图片
    x = tf.image.decode_jpeg(x, channels=3) # 图片解码
    x = tf.image.resize(x, [244, 244]) # 图片缩放

    ### 增强数据
    x = tf.image.random_flip_left_right(x) # 随机左右镜像
    x = tf.image.random_crop(x, [224, 224, 3]) # 随机裁剪

    # 转换成张量
    # x:[0, 255] -> 0~1
    x = tf.cast(x, dtype=tf.float32)/255.
    x = normalize(x) #标准化
    y = tf.convert_to_tensor(y)
    return x, y

# 这里的mean和std根据真实的数据计算获得，比如ImageNet
img_mean = tf.constant([0.485, 0.456, 0.406])
img_std = tf.constant([0.229, 0.224, 0.225])
def normalize(x, mean=img_mean, std=img_std):
    x = (x - mean)/std
    return x

def denormalize(x, mean=img_mean, std=img_std):
    x = x*std + mean
    return x

def show_pic(pic,name='img',cmap=None, deNorm=False):
    '''显示图像'''
    if  deNorm == True:
        pic = denormalize(pic)
    plt.imshow(pic,cmap=cmap)
    plt.axis('off')   # 打开坐标轴为 on
# 设置图像标题
    plt.title('%s'%(name))
    plt.show()

batchsz = 32
root = r'C:\Users\ThinkPad\Desktop\hymenoptera_data'
images, labels, table = load_insect(root, mode='train')
print('images:', len(images), images)
print('labels:', len(labels), labels)
print('table:', table)

db_train = tf.data.Dataset.from_tensor_slices((images, labels)).shuffle(1000).map(preprocess).batch(32)
images_val, labels_val, table = load_insect(root, mode='val')
db_test = tf.data.Dataset.from_tensor_slices((images_val, labels_val)).map(preprocess).batch(32)

# 采样一个样本
sample = next(iter(db_train))
print('sample:',sample[0].shape,'-->', sample[1].shape, 'min:',tf.reduce_min(sample[0]),'max:', tf.reduce_max(sample[0]))
table_reverse = dict(zip(table.values(), table.keys()))

print('###############################################')
# t = sample
# tt = t[1][0]
# tt = tt.numpy()
# print(table_reverse[tt])
# show_pic(t[0][0],name=table_reverse[t[1][0].numpy()],deNorm=True)
# for i,batch in enumerate(db_test):
#     for x,y in zip(batch[0], batch[1]):
#         show_pic(x, table_reverse[y.numpy()], deNorm=True)

# for a,b in zip(sample[0], sample[1]):
#     print(a,b)
    # show_pic(a, table_reverse[b.numpy()], deNorm=True)
    # for i in range(x_test.shape[0]):
    #     x_test[i] = denormalize(x_test[i])
    #     show_pic(x_test[i])



# 创建TensorBoard对象
# writter = tf.summary.create_file_writer('logs')
# for step, (x, y) in enumerate(db_train):
#     print(step,'-->','x:',x.shape,'y:',y.shape)
    # x: [32, 224, 224, 3]
    # y: [32]
    # with writter.as_default():
    #     x = denormalize(x)  # 反向normalize，方便可视化
    #     # 写入图片数据
    #     tf.summary.image('img', x, step=step, max_outputs=9)
    #     time.sleep(5)

from Example.Insect_CLF import ResNet
model = ResNet.resnet18()  # ResNet18网络
model0 = Sequential([
    layers.Flatten(),
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu ),
    layers.Dense(2, activation=tf.nn.softmax)
])
model.build(input_shape=(None, 224, 224, 3))
model.summary()  # 统计网络参数

optimizer = optimizers.Adam(lr=1e-4)
metric = metrics.CategoricalAccuracy()

for epoch in range(200):
    for step, (x,y) in enumerate(db_train):
        with tf.GradientTape() as tape:
            logits = model(x)

            y_onehot = tf.one_hot(y, depth=2)
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
            metric.update_state(y_onehot, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step % 16 == 0:
            print('epoch:',epoch, 'step:',step, 'loss:', float(loss), 'acc:', metric.result().numpy())
            metric.reset_states()

    # 测试验证
    metric = keras.metrics.CategoricalAccuracy()
    for x, y in db_test:
        # [b, 10]
        y = tf.one_hot(y, depth=2)
        logits = model.predict(x)
        # be careful, these functions can accept y as [b] without warnning.
        metric.update_state(y, logits)
    print('test acc:', metric.result().numpy())
    metric.reset_states()



    # test_acc = model.evaluate(db_test)
    # print('epoch:',epoch, 'test_acc:',test_acc)

    # for i, batch in enumerate(db_test):
    #     y_pre = model.predict(batch[0])
    #     print('y_pre:',[tf.argmax(x) for x in y_pre])
    #     print('y_tru:',batch[1])
        # for x, y in zip(batch[0], batch[1]):
            # show_pic(x, table_reverse[y.numpy()], deNorm=True)

    # for a, b in zip(sample[0], sample[1]):
    #     print(a, b)
    #     show_pic(a, table_reverse[b.numpy()], deNorm=True)

    # for t, (x_test, y_test) in enumerate(db_test):
    #     for i in range(x_test.shape[0]):
    #         x_test[i] = denormalize(x_test[i])
    #         show_pic(x_test[i])
        # print(x_test.shape)




