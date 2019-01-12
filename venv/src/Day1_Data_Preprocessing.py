# step1 导入需要的库
import numpy as np  # NumPy包含数学计算函数。
import pandas as pd  # Pandas用于导入和管理数据集。

# step2 导入数据集
dataset = pd.read_csv('../datasets/Data.csv')
#不包括最后一列
X = dataset.iloc[ : , :-1].values
#取最后一列，即第4列
Y = dataset.iloc[ : ,3].values
print("Step2 :Importing dataset")
print("X")
print(X)
print("Y")
print(Y)

# step3 处理丢失数据
from sklearn.preprocessing import Imputer
# axis=0表示按列进行
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
#处理的是第1到(3-1)，（X[: ,m:n]：取第m到(n-1)列数据）
imputer = imputer.fit(X[ : ,1 : 3])
X[ : , 1:3] = imputer.transform(X[ : ,1 : 3])
print("===========")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)

#step4 解析分类数据
#分类数据指的是含有标签值而不是数字值的变量。取值范围通常是固定的。
# 例如"Yes"和"No"不能用于模型的数学计算，所以需要解析成数字。
# 为实现这一功能，我们从sklearn.preprocessing库导入LabelEncoder类。
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print('--------------')
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

#step 5 拆分数据集为测试集合和训练集合
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
print('-=-=---=-=-=-==--=')
print("step5 Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

#step 6:特征缩放
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
print("**************")
print("Step6 Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)

