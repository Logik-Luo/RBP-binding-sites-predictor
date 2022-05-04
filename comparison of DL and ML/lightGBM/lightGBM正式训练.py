from sklearn.utils import resample
import numpy as np
import pandas as pd

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


import lightgbm as lgb

params = {
    'application': 'binary',
    'n_estimators': 500,        # boosting的迭代次数
    'learning_rate': 0.1,
    'num_leaves': 380,      # 一棵树上的叶子节点个数
    'lambda_l1': 0.0008898137729663044,     # 设置一个 threshold，gain 小于这个 threshold 直接认为是 0，不再分裂
    'lambda_l2': 1.2293743925788878e-06,        # 为 gain 的分母（即节点样本数）增加一个常数项，作用于全程，在节点样本数已经很小的时候，能显著减小 gain 避免分裂
    'feature_fraction': 0.7968924591990494,     # 指定训练每棵树时要采样的特征百分比
    'bagging_fraction': 0.8847881204293092,     # 不进行重采样的情况下随机选择部分数据
    'bagging_freq': 6,      # 每6轮迭代进行一次bagging
    'min_data_in_leaf': 20,     #
    'extra_trees': False,
    # 'max_depth': 5 ，别碰它...

}
# params = {
#     'n_estimators': 1657,
#     'learning_rate': 0.07663607371059684,
#     'num_leaves': 393,
#     'max_depth': 13,
#     'min_data_in_leaf': 12,
#     'max_bin': 38,
#     'lambda_l1': 1,
#     'lambda_l2': 11,
#     'min_gain_to_split': 0.07598699857183397,
#     'bagging_fraction': 0.7351485420657539,
#     'bagging_freq': 18,
#     'feature_fraction': 0.6093335495893886
# }

light = lgb.LGBMClassifier(**params)

light.fit(train_x, train_y)
# y_predict = light.predict(test_data)

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

y_pred = light.predict(test_data)
print('AUC:', roc_auc_score(test_labels, y_pred))
print('ACC:', accuracy_score(test_labels, y_pred))
print('F1 score:', f1_score(test_labels, y_pred))
print('precision:', precision_score(test_labels, y_pred))
print('recall:', recall_score(test_labels, y_pred))

