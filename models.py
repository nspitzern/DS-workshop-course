import json

import pickle

import torch

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

from xgboost import XGBRegressor

import numpy as np


def get_models(filepath='', is_basic=False):
    models = dict()
    
    if filepath != '':    
        with open(filepath, 'rb') as f:
            models = pickle.load(f)
            
            print('Models Loaded!')
    else:
        # create models
        print('Creating models...')
        reg_model = LinearRegression()
        
        models.update({
            'lin_reg': reg_model
        })
        
        print('Models Created!')
        
    return models

def get_models_results(df, target, models=None, test_size=0.2, ignore_columns=None, load_data=False, data_file_path='', save_models=False, is_basic=False, verbose=True):
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
        
        assert models is not None, 'Please provide models'

        # run each model
        print('Running models...')
        print('Running Linear Regression...')
        reg_model = models['lin_reg'].fit(X_train, y_train[target])
        
        if save_models:
            with open(f'{"basic_" if is_basic else ""}models.pickle', 'wb') as f:
                pickle.dump(models, f, protocol=pickle.HIGHEST_PROTOCOL)

        # evaluate results
        results_map = {
                'lin_reg': _get_scores(reg_model.predict, X_test, y_test, target)
        }
    
    return results_map

def _get_scores(prediction_func, X, y, target):
    y_hat = prediction_func(X)
    
    mape = mean_absolute_percentage_error(y_hat, y[target])
    rmse = mean_squared_error(y_hat, y[target], squared=False)
    mae = mean_absolute_error(y_hat, y[target])
    r2 = r2_score(y_hat, y[target])
    
    return {'MAPE': mape, 'RMSE': rmse, 'MAE': mae, 'R^2': r2}