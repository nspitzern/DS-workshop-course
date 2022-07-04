import json

import pickle

import torch

import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBClassifier, XGBRegressor

import numpy as np

from dnn import get_dnn_model, get_dnn_results, dnn_predict


def get_models(filepath='', dnn_dim=None, is_basic=False):
    models = dict()
    assert dnn_dim is not None, "Please provide in_dim for the dnn model (len(X_train.columns))"
    
    if filepath != '':    
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
            
            dnn_model = get_dnn_model(dnn_dim)
            dnn_model_path = f'{"basic_" if is_basic else ""}dnn_model.pth'
            dnn_model.load_state_dict(torch.load(dnn_model_path))
            
            models.update({
                'dnn_model': dnn_model
            })
            
            print('Models Loaded!')
    else:
        assert dnn_dim is not None, "Please provide in_dim for the dnn model (len(X_train.columns))"
        # create models
        print('Creating models...')
        reg_model = LinearRegression()
        reg_tree = DecisionTreeRegressor()
        forest_reg = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=5)
        ada_reg = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=50)
        xg_reg_model = XGBRegressor()
        dnn_model = get_dnn_model(dnn_dim)
        
        models.update({
            'lin_reg': reg_model,
            'reg_tree': reg_tree,
            'forest_reg': forest_reg,
            'ada_reg': ada_reg,
            'xg_reg_model': xg_reg_model,
            'dnn_model': dnn_model
        })
        
        print('Models Created!')
        
    return models

def get_models_results(df, target, models=None, test_size=0.2, ignore_columns=None, load_data=False, data_file_path='', verbose=True):
    if load_data:
        assert data_file_path != '', "Please provide path to the data"
        
        with open(data_file_path, 'r') as f:
            results_map = json.load(f)
    else:
        # split to train and test
        print('Splitting to Train/Test...')
        ignore_columns = ignore_columns if ignore_columns is not None else []
        
        y = df[[target]]
        X = df.drop([target] + ignore_columns, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        print(f'Train Size: X={X_train.shape}, Y={y_train.shape}')
        print(f'Test Size: X={X_test.shape}, Y={y_test.shape}')

        # run each model
        print('Running models...')
        print('Running Linear Regression...')
        reg_model = models['lin_reg'].fit(X_train, y_train[target])
        print('Running Decision Tree...')
        reg_tree = models['reg_tree'].fit(X_train, y_train[target])
        print('Running Random Forest...')
        forest_reg = models['forest_reg'].fit(X_train, y_train[target])
        print('Running AdaBoosting...')
        ada_reg = models['ada_reg'].fit(X_train, y_train[target])
        print('Running XGBoost')
        xg_reg_model = models['xg_reg_model'].fit(X_train, y_train[target])
        print('Running Neural Network...')
        dnn_model = models['dnn_model']
        train_losses, test_losses = get_dnn_results(X_train, X_test, y_train, y_test, dnn_model, verbose)

        # evaluate results
        results_map = {
                'lin_reg': _get_scores(reg_model, reg_model.predict, X_test, y_test, target),
                'reg_tree': _get_scores(reg_tree, reg_tree.predict, X_test, y_test, target),
                'forest_reg': _get_scores(forest_reg, forest_reg.predict, X_test, y_test, target),
                'ada_reg': _get_scores(ada_reg, ada_reg.predict, X_test, y_test, target),
                'xg_reg_model': _get_scores(xg_reg_model, xg_reg_model.predict, X_test, y_test, target),
                'dnn_model': _get_scores(dnn_model, dnn_predict, X_test, y_test, target, dnn=True)
        }
    
    return results_map

def _get_scores(model, prediction_func, X, y, target, dnn=False):
    if dnn:
        y_hat = prediction_func(model, X)
    else:
        y_hat = prediction_func(X)
    
    mse = mean_squared_error(y_hat, y[target])
    rmse = mean_squared_error(y_hat, y[target], squared=False)
    mae = mean_absolute_error(y_hat, y[target])
    r2 = r2_score(y_hat, y[target])
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R^2': r2}