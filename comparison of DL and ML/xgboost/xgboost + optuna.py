from sklearn.utils import resample
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

train = pd.read_csv("./ml.csv")
df_majority = train[train.label == 0]
df_minority = train[train.label == 1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=39325,    # to match majority class
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
train_x = df_upsampled.drop('label', axis=1)     # 删除表中的某一行或者某一列
train_y = df_upsampled['label']
Xy = train_test_split(train_x, train_y, test_size=0.2)
train_data, test_data, train_labels, test_labels = Xy

def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
    param = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        # this parameter means using the GPU when training our model to speedup the training process
        'eta': trial.suggest_float('eta', 0, 1),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        'subsample': trial.suggest_float('subsample', 0, 1),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
        'n_estimators': trial.suggest_int('n_estimators', 380, 4000),
        'max_depth': trial.suggest_int('max_depth', 8, 16),
        'random_state': trial.suggest_int('random_state', 1, 2000),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
    }
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100)
    y_preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_preds, squared=False)
    return rmse


study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(objective, n_trials=100)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)