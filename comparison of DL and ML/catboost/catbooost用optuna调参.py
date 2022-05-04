from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor
import optuna


def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2)
    param = {
        "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1),
        "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-2, 1),
        # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1),
        "depth": trial.suggest_int("depth", 1, 16),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
        "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 20),
        "iterations": trial.suggest_int("iterations", 500, 5000),
        "random_strength": trial.suggest_float("random_strength", 0.1, 50),
        "rsm": trial.suggest_float("rsm", 0, 1),
        # "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1)
    }
    # Conditional Hyper-Parameters
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

    reg = CatBoostRegressor(**param)
    reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100)
    y_pred = reg.predict(X_test)
    score = r2_score(y_test, y_pred)
    return score


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
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
# reg = CatBoostRegressor(**param)
# y_pred = reg.predict(test_data)
# print('AUC→',roc_auc_score(test_labels, y_pred))
# print('ACC→',accuracy_score(test_labels, y_pred))
# print('F1 score→',f1_score(test_labels, y_pred))
# print('precision→',precision_score(test_labels, y_pred))
# print('recall→',recall_score(test_labels, y_pred))

study = optuna.create_study(sampler=TPESampler(), direction="maximize")
study.optimize(objective, n_trials=100)  # Run for 10 minutes，timeout=600
print("Number of completed trials: {}".format(len(study.trials)))
print("Best trial:")
trial = study.best_trial

print("\tBest Score: {}".format(trial.value))
print("\tBest Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))