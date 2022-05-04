import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from model import FM, DNN
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics


if __name__=='__main__':
    import pandas as pd

    # file_path = pd.read_csv('./dl.csv')
    df = pd.read_csv('./dl.csv')

    from sklearn.utils import resample

    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=39325,  # to match majority class
                                     random_state=123)
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    train_x = df_upsampled.drop('label', axis=1)  # 删除表中的某一行或者某一列
    train_y = df_upsampled['label']

    X = train_x
    Y = train_y

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    scaler = StandardScaler()  # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
    X_train = scaler.fit_transform(X_train)  # 计算训练集的平均值和标准差，以便测试数据集使用相同的变换
    X_test = scaler.transform(X_test)

    y_train = Y_train.to_numpy()
    y_test = Y_test.to_numpy()

    # X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Adding a third dimension to our data
    # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    k = 32
# **************** Statement 1 of Training *****************#
    model = FM(k)
    optimizer = optimizers.Adam(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(65536).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=200)

    # 评估
    fm_pre = model(X_test)
    fm_pre = [1 if x > 0.5 else 0 for x in fm_pre]

# **************** Statement 2 of Training *****************#
    # 获取FM训练得到的隐向量
    v = model.variables[1]  # [None, onehot_dim, k] , 获取按范围和/或后缀过滤的模型变量列表

    X_train = tf.cast(tf.expand_dims(X_train, -1), tf.float32)  # [None, onehot_dim, 1]  ,张量数据类型转换
    X_train = tf.reshape(tf.multiply(X_train, v), shape=(-1, v.shape[0]*v.shape[1]))  # [None, onehot_dim*k]

    hidden_units = [1024, 512, 128, 32, 8]
    model = DNN(hidden_units, 1, 'relu')
    optimizer = optimizers.Adam(0.01)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = train_dataset.batch(65536).prefetch(tf.data.experimental.AUTOTUNE)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=1000)

    # 评估
    X_test = tf.cast(tf.expand_dims(X_test, -1), tf.float32)
    X_test = tf.reshape(tf.multiply(X_test, v), shape=(-1, v.shape[0]*v.shape[1]))
    fnn_pre = model(X_test)
    fnn_pre = [1 if x > 0.5 else 0 for x in fnn_pre]

    Y_pred = model.predict(X_test)
    Y_pred = np.where(Y_pred >= 0.5, 1, 0)  # 查找符合条件的位置信息，满足≥0.5返回1，否则0
    from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, accuracy_score, \
        roc_auc_score

    print(classification_report(Y_test, Y_pred))
    actual_acc = accuracy_score(Y_test, Y_pred)
    actual_rec = recall_score(Y_test, Y_pred)
    actual_p = precision_score(Y_test, Y_pred)
    actual_f1 = f1_score(Y_test, Y_pred)
    print('AUC ->', roc_auc_score(Y_test, Y_pred))
    print('Recall ->', actual_rec)
    print('Precision ->', actual_rec)
    print('F1 Score ->', actual_f1)
    print('Accuracy ->', actual_acc)

