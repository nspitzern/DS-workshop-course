import sklearn
from sklearn.linear_model import LinearRegression, LogisticRegression, ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBClassifier, XGBRegressor

from dnn import get_dnn_model, get_dnn_results, dnn_predict

def get_models_results(df, target, test_size=0.2, verbose=True):
    # split to train and test
    print('Splitting to Train/Test...')
    y = df[[target]]
    X = df.drop([target], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f'Train Size: X={X_train.shape}, Y={y_train.shape}')
    print(f'Test Size: X={X_test.shape}, Y={y_test.shape}')
    
    # create models
    print('Creating models...')
    reg_model = LinearRegression()
    reg_tree = DecisionTreeRegressor()
    forest_reg = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=5)
    ada_reg = AdaBoostRegressor(DecisionTreeRegressor(), n_estimators=50)
    xg_reg_model = XGBRegressor()
    dnn_model = get_dnn_model(len(X_train.columns))
    
    
    # run each model
    print('Running models...')
    reg_model = reg_model.fit(X_train, y_train[target])
    reg_tree = reg_tree.fit(X_train, y_train[target])
    forest_reg = forest_reg.fit(X_train, y_train[target])
    ada_reg = ada_reg.fit(X_train, y_train[target])
    xg_reg_model = xg_reg_model.fit(X_train, y_train[target])
    train_losses, test_losses = get_dnn_results(X_train, X_test, y_train, y_test, dnn_model, verbose=False)
    
    # evaluate results
    results_map = {
            'lin_reg': _get_scores(reg_model.predict, X_test, y_test, target),
            'reg_tree': _get_scores(reg_tree.predict, X_test, y_test, target),
            'forest_reg': _get_scores(forest_reg.predict, X_test, y_test, target),
            'ada_reg': _get_scores(ada_reg.predict, X_test, y_test, target),
            'xg_reg_model': _get_scores(xg_reg_model.predict, X_test, y_test, target),
            'dnn_model': _get_scores(dnn_predict, X_test, y_test, target),
    }
    
    print(results_map)

def _get_scores(prediction_func, X, y, target):
    y_hat = prediction_func(X)
    
    mse = mean_squared_error(y_hat, y[target])
    rmse = mean_squared_error(y_hat, y[target], squared=False)
    mae = mean_absolute_error(y_hat, y[target])
    r2 = r2_score(y_hat, y[target])
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R^2': r2}