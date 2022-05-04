import optuna
import pandas as pd
from optuna.integration import LightGBMPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import lightgbm as lgbm

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
from optuna.samplers import TPESampler

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
# Xy = train_test_split(train_x, train_y, test_size=0.2)
# train_data, test_data, train_labels, test_labels = Xy


def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 10000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 1000),
        "max_bin": trial.suggest_int("max_bin", 10, 300),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.2, 0.95),
        "bagging_freq": trial.suggest_int("bagging_freq", 0, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 0.95),
    }
    # model = lgbm.LGBMClassifier(objective="binary", **params)
    # model.fit(
    #     X_train,
    #     y_train,
    #     eval_set=[(X_test, y_test)],
    #     eval_metric="binary_logloss",
    #     callbacks=[
    #         LightGBMPruningCallback(trial, "binary_logloss")
    #     ],
    # )
    model = lgbm.LGBMClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    return score

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, r2_score

# y_pred = model.predict(test_data)
# print('AUC→',roc_auc_score(test_labels, y_pred))
# print('ACC→',accuracy_score(test_labels, y_pred))
# print('F1 score→',f1_score(test_labels, y_pred))
# print('precision→',precision_score(test_labels, y_pred))
# print('recall→',recall_score(test_labels, y_pred))
#
# study = optuna.create_study(direction="minimize", study_name="LGBM Classifier")
# func = lambda trial: objective(trial)
# study.optimize(func, n_trials=100)
study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100)  # Run for 10 minutes，timeout=600
print("Number of completed trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("\tBest Score: {}".format(trial.value))
print("\tBest Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))