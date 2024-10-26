# import libraries
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from category_encoders import OneHotEncoder
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.linear_model import LinearRegression
import model_performance
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.feature_selection import RFE


# split data
def split_data(data, target, size, randomstate):
    '''
    The function splits data into train and test sets
    '''
    X = data.drop(target, axis=1)
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=size, 
        random_state=randomstate
    )

    return X_train, X_test, y_train, y_test

# Function to process data for linear regression
def process_data_for_LR(X_train, X_test, y_train, y_test):
    '''
    Processes data for linear regression modelling

    '''
    cols_to_drop = [
    'children',
    'region',
    'sex'
    ]
    X_train.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)
    ohe = OneHotEncoder(use_cat_names=True)
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)

    cols_to_drop = ['smoker_no']
    X_train.drop(cols_to_drop, axis=1, inplace=True)
    X_test.drop(cols_to_drop, axis=1, inplace=True)

    # transform
    pt = PowerTransformer(method='yeo-johnson')
    y_train_t = pt.fit_transform(y_train.values.reshape(-1, 1))[:, 0]
    y_test_t = pt.transform(y_test.values.reshape(-1, 1))[:, 0]

    return X_train, X_test, y_train, y_test, y_train_t, y_test_t, pt



# Function to train Linear Regression
def train_and_evaluate_LR(X_train, X_test, y_train, y_test, y_train_t, y_test_t, pt):
    '''
    The function trains linear regression model on the processed data
    '''
    sample_weight = y_train / y_train.min()

    lr = LinearRegression()
    lr.fit(
        X_train, 
        y_train_t, 
        sample_weight=sample_weight
    )
    # evaluate
    y_pred_train = lr.predict(X_train)
    y_pred_test = lr.predict(X_test)

    y_pred_train = pt.inverse_transform(y_pred_train.reshape(-1, 1))[:, 0]
    y_pred_test = pt.inverse_transform(y_pred_test.reshape(-1, 1))[:, 0]

    base_perf_train = model_performance.calc_model_performance(y_train, y_pred_train)
    base_perf_test = model_performance.calc_model_performance(y_test, y_pred_test)

    print('Linear Regression Results for Training set')
    print(base_perf_train)
    print(" ")
    print('Linear Regression Results for Testing set')
    print(base_perf_test)

    return y_pred_train, y_pred_test, lr

# function to process data for xgboost modelling
def process_data_for_xgboost(X_train, X_test):
    '''
    The function processes data for xgboost modelling.
    '''
    ohe = OneHotEncoder(use_cat_names=True)
    X_train = ohe.fit_transform(X_train)
    X_test = ohe.transform(X_test)

    return X_train, X_test, ohe

    
    
    
def train_and_evaluate_xgboost(X_train, X_test, y_train, y_test): 
    '''
    The function trains and evaluates xgboost model using bayesian search.
    '''   
    rfe = RFE(estimator=XGBRegressor())
    xgb = XGBRegressor()

    steps = [
    ('rfe', rfe),
    ('xgb', xgb)
    ]

    pipe = Pipeline(steps)

    num_features = X_train.shape[1]
    search_spaces = {
        'rfe__n_features_to_select': Integer(1, num_features), # Num features returned by RFE
        'xgb__n_estimators': Integer(1, 500), # Num trees built by XGBoost
        'xgb__max_depth': Integer(2, 8), # Max depth of trees built by XGBoost
        'xgb__reg_lambda': Integer(1, 200), # Regularisation term (lambda) used in XGBoost
        'xgb__learning_rate': Real(0, 1), # Learning rate used in XGBoost
        'xgb__gamma': Real(0, 2000) # Gamma used in XGBoost
    }

    xgb_bs_cv = BayesSearchCV(
    estimator=pipe, # Pipeline
    search_spaces=search_spaces, # Search spaces
    scoring='neg_root_mean_squared_error', # BayesSearchCV tries to maximise scoring metric, so negative RMSE used
    n_iter=75, # Num of optimisation iterations
    cv=3, # Number of folds
    n_jobs=-1, # Uses all available cores to compute
    verbose=1, # Show progress
    random_state=0 # Ensures reproducible results
    )


    xgb_bs_cv.fit(
    X_train, 
    y_train,
    )

    y_pred_train_xgb = xgb_bs_cv.predict(X_train)
    y_pred_test_xgb = xgb_bs_cv.predict(X_test)

    xgb_perf_train = model_performance.calc_model_performance(y_train, y_pred_train_xgb)

    xgb_perf_test = model_performance.calc_model_performance(y_test, y_pred_test_xgb)

    print('XGBoost Results for Training set')
    print(xgb_perf_train)
    print(" ")
    print('XGBoost Results for Testing set')
    print(xgb_perf_test)

    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_bs_cv, f)
    print("XGBoost model saved!")

    return y_pred_train_xgb, y_pred_test_xgb, xgb_bs_cv


