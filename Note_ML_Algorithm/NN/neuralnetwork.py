# Tensorflow-GPU 2.1版本
# from keras import layers
# from keras import models

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split as TTS, cross_val_score as CVS, StratifiedKFold as SKF, KFold
from sklearn.metrics import r2_score

from Note_ML_Algorithm.EvaluationModel import EvaluationModel as EM
import datetime
import numpy as np
import os


# NN_cpu
def Network_cpu(Xtrain, Ytrain, epochs=5, nOutput=1, save=0, cv=0):
    start = datetime.datetime.now()
    X_train,X_test,Y_train,Y_test = TTS(Xtrain, Ytrain,test_size=0.1)
    network = models.Sequential()
    network.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1], )))
    network.add(layers.Dense(64, activation='relu'))
    network.add(layers.Dense(nOutput))
    network.compile(optimizer='rmsprop', loss='mae', metrics=['accuracy'])  # 均方误差作为损失
    print(network.summary())


    if cv != 0:
        r2=[]
        kfold = KFold(n_splits=cv,shuffle=True,random_state=1)
        for train, test in kfold.split(Xtrain,Ytrain):
            network.fit(Xtrain[train], Ytrain[train],epochs=epochs)
            r2.append(r2_score(Ytrain[test], network.predict(Xtrain[test])))

        #scores = CVS(network, Xtrain, Ytrain, cv=10, scoring='neg_root_mean_squared_error')
        print(r2, np.mean(r2))


    network.fit(X_train, Y_train, epochs=epochs,
                          batch_size=1, verbose=1,
                          validation_data=(X_test, Y_test))


    end = datetime.datetime.now()

    #模型评估
    # EM.modelEvaluate(network, X_test, Y_test)
    r2 = r2_score(Y_test,network.predict(X_test))
    print('# # # r2 # # #:', r2)

    if save:
        modelName = 'nn_regr_'+end.strftime('%m%d_%H%M%S')+'.h5'
        modelFile = r'../module_ML/saved_model/'+modelName
        network.save(modelFile)
        # joblib.dump(regr, modelFile)
        print('Model save to:',modelFile)
    else:
        pass
    print('Neural Network training was successful!\nTotal time :', (end - start))

    return network



# NN gpu
def Network_gpu(Xtrain,Ytrain,epochs=100,nOutput=1,save=0):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
    # 获取所有GPU 设备列表
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置GPU 显存占用为按需分配，增长式
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # 异常处理
            print(e)

    start = datetime.datetime.now()
    X_train,X_test,Y_train,Y_test = TTS(Xtrain,Ytrain,test_size=0.2)
    network = models.Sequential()
    network.add(layers.Dense(64, activation='relu', input_shape=(X_train.shape[1], )))
    network.add(layers.Dense(32, activation='relu'))
    network.add(layers.Dense(nOutput))
    network.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])  # 均方误差作为损失
    print(network.summary())

    network.fit(X_train, Y_train, epochs=epochs,
                          batch_size=1, verbose=1,
                          validation_data=(X_test, Y_test))



    end = datetime.datetime.now()

    if save:
        modelName = 'nn_regr_'+end.strftime('%m%d_%H%M%S')+'.h5'
        modelFile = r'../module_ML/saved_model/'+modelName
        network.save(modelFile)
        # joblib.dump(regr, modelFile)
        print('Model save to:',modelFile)
    else:
        pass
    print('Neural Network training was successful!\nModel save to: ../module_ML/saved_model/'+modelName+'.h5\nTotal time :',(end - start))

    return network
