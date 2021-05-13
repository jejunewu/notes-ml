import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets, Sequential, losses,optimizers
import matplotlib.pyplot as plt

# 加载数据
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
print(x_train.shape,'===>', y_train.shape)
print(x_test.shape,'===>', y_test.shape)

# 归一化
x_train = x_train/255.
x_test = x_test/255.
y_train = tf.squeeze(y_train,axis=1)
y_test  = tf.squeeze(y_test, axis=1)
print(y_train.shape)
db_train = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32).shuffle(32)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

from Example.CIFAR10.network import VGG16

model = VGG16()
model.build(input_shape=(32, 32, 32, 3))
model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# model.compile(optimizer=keras.optimizers.Adam(0.001),
#                   loss=keras.losses.CategoricalCrossentropy(from_logits=True),
#                   metrics=['accuracy'])
model.fit(db_train, epochs=10, batch_size=50, validation_data=db_test)

# must specify from_logits=True!
# criteon = keras.losses.CategoricalCrossentropy(from_logits=True)
# metric = keras.metrics.CategoricalAccuracy()
#
# optimizer = optimizers.Adam(learning_rate=0.0001)
#
# for epoch in range(250):
#
#     for step, (x, y) in enumerate(db_train):
#         # [b, 1] => [b]
#         y = tf.squeeze(y, axis=1)
#         # [b, 10]
#         y = tf.one_hot(y, depth=10)
#
#         with tf.GradientTape() as tape:
#             logits = model(x)
#             loss = criteon(y, logits)
#             # loss2 = compute_loss(logits, tf.argmax(y, axis=1))
#             # mse_loss = tf.reduce_sum(tf.square(y-logits))
#             # print(y.shape, logits.shape)
#             metric.update_state(y, logits)
#
#         grads = tape.gradient(loss, model.trainable_variables)
#         # MUST clip gradient here or it will disconverge!
#         grads = [tf.clip_by_norm(g, 15) for g in grads]
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#         if step % 40 == 0:
#             # for g in grads:
#             #     print(tf.norm(g).numpy())
#             print(epoch, step, 'loss:', float(loss), 'acc:', metric.result().numpy())
#             metric.reset_states()
#
#     if epoch % 1 == 0:
#
#         metric = keras.metrics.CategoricalAccuracy()
#         for x, y in db_test:
#             # [b, 1] => [b]
#             y = tf.squeeze(y, axis=1)
#             # [b, 10]
#             y = tf.one_hot(y, depth=10)
#
#             logits = model.predict(x)
#             # be careful, these functions can accept y as [b] without warnning.
#             metric.update_state(y, logits)
#         print('test acc:', metric.result().numpy())
#         metric.reset_states()





