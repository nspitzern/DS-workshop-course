import json

import pickle

import torch

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import numpy as np


def get_models(filepath='', verbose=False):
    models = dict()
    
    if filepath != '':    
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
            
            if verbose:
                print('Model Loaded!')
    else:
        # create models
        if verbose:
            print('Creating models...')
        model = LinearRegression()
        
        if verbose:
            print('Model Created!')
        
    return model

def get_models_results(df, target, model=None, test_size=0.2, ignore_columns=None, load_data=False, data_file_path='', save_models=False, is_basic=False, verbose=True):
    if load_data:
        assert data_file_path != '', "Please provide path to the data"
        
        with open(data_file_path, 'r') as f:
            results_map = json.load(f)
    else:
        # split to train and test
        if verbose:
            print('Splitting to Train/Test...')
        ignore_columns = ignore_columns if ignore_columns is not None else []
        
        y = df[target]
        X = df.drop([target] + ignore_columns, axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        if verbose:
            print(f'Train Size: X={X_train.shape}, Y={y_train.shape}')
            print(f'Test Size: X={X_test.shape}, Y={y_test.shape}')
        
        assert model is not None, 'Please provide model'

        # run each model
        if verbose:
            print('Running model...')
        model.fit(X_train, y_train)
        
        if save_models:
            with open(f'{"basic_" if is_basic else ""}model.pickle', 'wb') as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # evaluate results
        results_map = _get_scores(model.predict, X_test, y_test)
    
    return results_map

def _get_scores(prediction_func, X, y):
    y_hat = prediction_func(X)
    
    rmse = mean_squared_error(y, y_hat, squared=False)
    mae = mean_absolute_error(y, y_hat)
    r2 = r2_score(y, y_hat)
    
    return {'RMSE': rmse, 'MAE': mae, 'R^2': r2}