
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF

from sklearn.datasets import load_boston

from sklearn.model_selection import KFold, cross_val_score as CVS , train_test_split as TTS
from ML_Algorithm_test.EvaluationModel import EvaluationModel as EM

import matplotlib.pyplot as plt
import numpy as np


# 1 准备数据
trainingDataHome = '../TrainingData/'
x = np.load(trainingDataHome+'training_inputs.npy')

y = np.load(trainingDataHome+'training_outputs.npy')

print('all:',x.shape,'--->',y.shape)

# 2 分割训练数据和测试数据
# 随机采样25%作为测试 75%作为训练
x_train, x_test, y_train, y_test = TTS(x, y, test_size=0.1, random_state=33)

print('train:',x_train.shape,'--->',y_train.shape)
print('test:',x_test.shape,'--->',y_test.shape)


# kernel = C(constant_value=0.2, constant_value_bounds=(1e1, 1e4)) * RBF(length_scale=0.5, length_scale_bounds=(1e1, 1e4))
# kernel = RBF(length_scale=0.1,length_scale_bounds=(1e1, 1e3))
# kernel = RBF(length_scale=0.5, length_scale_bounds=(1e-4, 1e4))
# kernel = RBF(length_scale=0.1, length_scale_bounds=(1e2, 1e4))

# gpr = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)

# mixed_kernel = kernel = C(1.0, (1e-4, 1e4)) * RBF(0.1, (1e-4, 1e4))
gpr = GPR(alpha=5,n_restarts_optimizer=20,kernel = mixed_kernel)

gpr.fit(x_train, y_train)

EM.modelEvaluate(gpr,x_test, y_test)

mu, cov = gpr.predict(x_test[:10], return_cov=True)
print('mu:',mu.shape,'cov:',cov.shape)


Ytest = mu.ravel()
# print(Ytest)
uncertainty = 1.96 * np.sqrt(np.diag(cov))
# print(uncertainty)


# plt.figure()
# # plt.title("l=%.2f sigma_f=%.2f" % (gpr.kernel_.k2.length_scale, gpr.kernel_.k1.constant_value))
# # plt.fill_between(Xtest.ravel(), Ytest + uncertainty, Ytest - uncertainty, alpha=0.1)
# plt.plot(x_test, y_test, label="predict")
# plt.scatter(x_train, y_train, label="train", c="red", marker="x")
# plt.legend()
# plt.show()
