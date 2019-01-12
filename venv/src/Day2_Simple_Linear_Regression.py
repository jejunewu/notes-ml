#数据预处理
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("../datasets/studentscores.csv")
X = dataset.iloc[ : , :1].values
Y = dataset.iloc[ : ,1].values

from sklearn.model_selection import train_test_split
     #随机划分测试集和训练集
        #train_data：所要划分的样本特征集
        #train_target：所要划分的样本结果
        #test_size：样本占比，如果是整数的话就是样本的数量
        #random_state：是随机数的种子。其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
        #比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。
        #但填0或不填，每次都会不一样。
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/4,random_state=0)

#训练集使用简单线性回归模型来训练
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train,Y_train)

#预测结果
Y_pred = regressor.predict(X_test)

#step4 可视化
    #训练集结果可视化
plt.scatter(X_train,Y_train,color = 'red')
plt.plot(X_train,regressor.predict(X_train),color = 'blue')
plt.show()

     #测试集结果可视化
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_test,regressor.predict(X_test),color = 'blue')
plt.show()

'''
print('X')
print(X)
print('Y')
print(Y)
'''