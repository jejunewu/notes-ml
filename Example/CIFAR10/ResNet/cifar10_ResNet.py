import tensorflow as tf
import os
from tensorflow.keras import datasets,optimizers,metrics
from Example.CIFAR10.ResNet import ResNet

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)

def preprocess(x, y):
    x = 2*tf.cast(x, dtype=tf.float32)/255. - 1
    y = tf.cast(y, dtype=tf.int32)
    return x, y

(x_train, y_train),(x_test, y_test) = datasets.cifar10.load_data()
y_train = tf.squeeze(y_train, axis=1)
y_test = tf.squeeze(y_test, axis=1)

print(x_train.shape,'-->',y_train.shape)
print(x_test.shape,'-->',y_test.shape)


# 训练集
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(1000).map(preprocess).batch(512)
# 测试集
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(preprocess).batch(512)

# 采样一个样本
sample = next(iter(train_db))
print('sample',sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))





# 训练
# [b, 32, 32, 3] => [b, 1, 1, 512]
model = ResNet.resnet18()  # ResNet18网络
model.build(input_shape=(None, 32, 32, 3))
model.summary()  # 统计网络参数
# model.compile(optimizer=optimizers.Adam(lr=1e-4), loss=tf.losses.sparse_categorical_crossentropy, metrics=['accuracy'])
# model.fit(train_db, epochs=100,validation_data=test_db,batch_size=512)

optimizer = optimizers.Adam(lr=1e-4)  # 构建优化器
metric = metrics.CategoricalAccuracy()

for epoch in range(100):
    for step, (x,y) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 10],前向传播
            logits = model(x)
            # [b] => [b, 10],one-hot编码
            y_onehot = tf.one_hot(y, depth=10)
            # 计算交叉熵
            loss = tf.losses.categorical_crossentropy(y_onehot, logits,from_logits=True)
            loss = tf.reduce_mean(loss)
            metric.update_state(y_onehot, logits)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if step%50 == 0:
            print(epoch,step,'loss:', float(loss), 'acc:',metric.result().numpy())
            metric.reset_states()

    # 测试集精度
    total_num = 0
    total_correct = 0
    for x,y in test_db:
        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct/total_num
    print(epoch, 'acc: ', acc)