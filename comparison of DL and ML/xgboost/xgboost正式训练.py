from sklearn.utils import resample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


train = pd.read_csv("./ml.csv")
df_majority = train[train.label==0]
df_minority = train[train.label==1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=39325,    # to match majority class
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
train_x = df_upsampled.drop('label', axis = 1)     # 删除表中的某一行或者某一列
train_y = df_upsampled['label']
Xy = train_test_split(train_x, train_y, test_size=0.2)
train_data, test_data, train_labels, test_labels = Xy

params = {
    'eta': 0.9079410979267565,
    'lambda': 7.650200254479571,
    'alpha': 0.007941182547432644,
    'colsample_bytree': 0.9826835296296769,
    'subsample': 0.7343542917805344,
    'learning_rate': 0.05239880634573202,
    'n_estimators': 3634,
    'max_depth': 16,
    # 'random_state': 1965,
    'min_child_weight': 1,
}


import xgboost as xgb
model = xgb.XGBClassifier(**params)
model.fit(train_data, train_labels, eval_set=[(test_data, test_labels)], verbose=0, early_stopping_rounds=20)
preds_raw = model.predict(test_data)
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
y_pred = model.predict(test_data)
print('AUC→',roc_auc_score(test_labels, y_pred))
print('ACC→',accuracy_score(test_labels, y_pred))
print('F1 score→',f1_score(test_labels, y_pred))
print('precision→',precision_score(test_labels, y_pred))
print('recall→',recall_score(test_labels, y_pred))