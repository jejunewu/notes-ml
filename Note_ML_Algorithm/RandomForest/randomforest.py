from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split as TTS
from sklearn.model_selection import cross_val_score as CVS
from Note_ML_Algorithm.EvaluationModel import EvaluationModel as EM
import datetime
import joblib

def model_RandomForest(X_train, Y_train, nTrees=100, cv=0, save=0, saveModelHome='./'):

    Xtrain,Xtest,Ytrain,Ytest = TTS(X_train,Y_train,test_size=0.2)

    regr = RandomForestRegressor(n_estimators = nTrees, criterion='mse',
        max_depth=None, min_samples_split=2, min_samples_leaf=1,
        min_weight_fraction_leaf=0.0, max_features = 3,
        max_leaf_nodes=None, min_impurity_decrease=0.0,
        min_impurity_split=None, bootstrap=True,
        oob_score=False, n_jobs=-1, random_state=None,
        verbose=0, warm_start=False)


    start = datetime.datetime.now()
    regr.fit(Xtrain, Ytrain)
    if cv != 0:
        print('交叉验证开始...')
        # regr.fit(X_train,Y_train)
        score = CVS(regr,X_train,Y_train,cv=cv,scoring='neg_root_mean_squared_error')
        print(score, score.mean())
        print(cv,'折交叉验证结束！')

    end = datetime.datetime.now()

    print('Random Forest training was successful!\nTotal time :',(end - start))

    #模型评估
    EM.modelEvaluate(regr, Xtest, Ytest)

    if save:
        modelName = 'rf_regr_'+end.strftime('%m%d_%H%M%S')+'.pkl'
        modelFile = saveModelHome + modelName
        joblib.dump(regr, modelFile)
        print('Model save to:', modelFile)
    else:
        pass
    return regr




# def model_RandomForest_2(X_train, Y_train,nTrees=1000,maxFeatures='log2'):
#
#     Xtrain,Xtest,Ytrain,Ytest = train_test_split(X_train,Y_train,test_size=0.3)
#
#     regr = RandomForestRegressor(n_estimators = nTrees,max_features=maxFeatures)
#
#
#     start = datetime.datetime.now()
#     regr.fit(Xtrain, Ytrain)
#     # score = cross_val_score(regr,Xtrain,Ytrain,cv=10).mean()
#     score = regr.score(Xtest,Ytest)
#     end = datetime.datetime.now()
#
#     print('Random Forest training was successful!\nTotal time :',(end - start),'\nScore : ',score)
#
#     return regr
