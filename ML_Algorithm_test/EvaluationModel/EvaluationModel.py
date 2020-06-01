from sklearn.metrics import r2_score as r2, mean_squared_error as MSE, mean_absolute_error as MAE
from sklearn.preprocessing import StandardScaler

def modelEvaluate(model, x_test,  y_test, ):

    y_predict = model.predict(x_test)
    # ss_y = StandardScaler()

    modelName=''
    for i in range(len(str(model))):
        if str(model)[i] == '(':
            modelName = str(model)[:i]
            break

    print('===============', modelName ,'===============')

    print("默认评估值为：", model.score(x_test, y_test))

    print("R^2值为：", r2(y_test, y_predict))

    # print("均方误差(MSE)为:", MSE(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))

    print("均方误差(MSE)为:", MSE(y_test, y_predict))

    # print("平均绝对误差(MAE)为:", MAE(ss_y.inverse_transform(y_test),ss_y.inverse_transform(y_predict)))
    print("平均绝对误差(MAE)为:", MAE(y_test, y_predict))

    print('='*(len(modelName)+32))

