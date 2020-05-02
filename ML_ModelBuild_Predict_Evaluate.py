from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
linear_regressor = LinearRegression()
DT_regressor = DecisionTreeRegressor()

def perform_LinearRegression(df_preprocessed,X,X_train, X_test, y_train, y_test):
    #linear_regressor = LinearRegression()
    linear_regressor.fit(X_train, y_train)
    print("Execute linear regression")
    print(X_test)
    mlr_pred = linear_regressor.predict(X_test)
    print('\n *****Multiple Linear Regression Results*****')
    print('\n prediction of multiple linear regression model for test data: ',mlr_pred)
    # The coefficients and intercept:
    print('\n Coefficients of multiple linear regression: ',linear_regressor.coef_)
    print('\n Intercept of multiple linear regression: ',linear_regressor.intercept_)
    # The mean squared error
    print('\n Mean squared error of multiple linear regression: %.2f ' % np.mean((linear_regressor.predict(X_test) - y_test) ** 2))
    # Explained coefficient of determination score: 1 is perfect prediction
    print('\n Coeff. determination score of multiple linear regression: %.2f' % linear_regressor.score(X_test, y_test))
    df_preprocessed['MLR_Prediction']=linear_regressor.predict(X)
    return df_preprocessed

def perform_DecisionTreeRegression(DT_result,df_preprocessed,X,X_train, X_test, y_train, y_test):
    #DT_regressor = DecisionTreeRegressor()
    DT_regressor.fit(X_train, y_train)
    # X_train=X_train.iloc[:, 1:21]
    dt_pred = DT_regressor.predict(X_test)
    print('\n *****Decision Tree Regression Results*****')
    print('\n prediction of decision tree regression model for test data: ',dt_pred)
    # The mean squared error
    print('\n Mean squared error of decision tree regression: %.2f ' % np.mean((DT_regressor.predict(X_test.values) - y_test.values) ** 2))
    # Explained coefficient of determination score: 1 is perfect prediction
    print('\n Coeff. determination score of decision tree  regression: %.2f' % DT_regressor.score(X_test.values, y_test.values))
    DT_result['DT_Prediction']=DT_regressor.predict(X)
    # save result in file
    filename = 'Crop_yield_predicted_output.csv'
    DT_result.to_csv(filename, index=False, encoding='utf-8')
    return DT_result

def perform_Machinelearningtasks(df_preprocessed):
    feature_cols = list(df_preprocessed)
    predict_cols = feature_cols[-1:]

    y = df_preprocessed[predict_cols]
    X = df_preprocessed.iloc[:,1:21]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    print(X_train)
    X_train.to_csv("C:/Users/i308124/Desktop/babu project/X_train.csv", index=False)
    X_test.to_csv("C:/Users/i308124/Desktop/babu project/X_test.csv", index=False)

    X_train=X_train.apply(pd.to_numeric, errors='coerce')
    X_train.fillna(0, inplace=True)
    X_test = X_test.apply(pd.to_numeric, errors='coerce')
    X_test.fillna(0, inplace=True)
    y_train = y_train.apply(pd.to_numeric, errors='coerce')
    y_train.fillna(0, inplace=True)
    y_test = y_test.apply(pd.to_numeric, errors='coerce')
    y_test.fillna(0, inplace=True)

    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)

    MLR_result=perform_LinearRegression(df_preprocessed,X,X_train, X_test, y_train, y_test)
    print('*****Displaying first five rows of Multiple linear regression output with input features')
    print(MLR_result.head())
    DT_result=perform_DecisionTreeRegression(MLR_result,df_preprocessed,X,X_train, X_test, y_train, y_test)
    print('*****Displaying first five rows of Decision tree regression output with input features')
    print(DT_result.head())