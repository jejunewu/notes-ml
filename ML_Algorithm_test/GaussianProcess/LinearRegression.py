# coding:utf-8  
'''
Created on 2019年9月27日

@author: Jason.Fang   11949039@mail.sustech.edu.cn
'''
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import GridSearchCV
from ML_Algorithm_test.EvaluationModel import EvaluationModel as EM

if __name__ == "__main__":
    
    #load boston dataset
    boston = load_boston()
    X = boston.data #get the feature
    y = boston.target #get the target vector
    X = X[y<50.0]#leave out outliers
    y = y[y<50.0]#leave out outliers
    print('Train Dataset Statistics: line = %d, row = %d'%(X.shape[0],X.shape[1]))
    
    #split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
    
    #training linear Regression
    lin_reg = LinearRegression()#default parameters
    param_test = {'fit_intercept':('True', 'False'), 'normalize':('True', 'False'), 'copy_X':('True', 'False')}
    gsearch = GridSearchCV( estimator=lin_reg , param_grid = param_test, cv=5)
    gsearch.fit(X_train, y_train)
    print("Testing Score@LR:{}".format(gsearch.score(X_test, y_test)))
    
    
    #training polynomial regression
    #poly_reg = Pipeline([("poly", PolynomialFeatures(degree=2)),("std_scaler", StandardScaler()),("lin_reg", LinearRegression())])
    poly_reg = Pipeline([("poly", PolynomialFeatures(degree=2)),("std_scaler", StandardScaler()),("lin_reg", Ridge())])#l2 regularization
    poly_reg.fit(X_train, y_train)
    print("Testing Score@Poly+Ridge:{}".format(poly_reg.score(X_test, y_test)))
    
    
    #training GaussianProcessRegressor 
    # 核函数的取值
    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))
    reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1)
    reg.fit(X_train, y_train)
    print("Testing Score@Gaussian:{}".format(reg.score(X_test, y_test)))

    EM.modelEvaluate(reg, X_test, y_test)
    
    

'''
Train Dataset Statistics: line = 490, row = 13
Testing Score@LR:0.7609245931839279
Testing Score@Poly+Ridge:0.8900214567939478
Testing Score@Gaussian:0.8900214567939478
'''
