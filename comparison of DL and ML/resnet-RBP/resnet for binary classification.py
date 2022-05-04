import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Conv1D, BatchNormalization, Activation, MaxPool1D, Dropout, Flatten, Dense
from tensorflow.keras import Model
import pandas as pd
df = pd.read_csv('./dl.csv')

from sklearn.utils import resample
df_majority = df[df.label==0]
df_minority = df[df.label==1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,     # sample with replacement
                                 n_samples=39325,    # to match majority class
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
train_x = df_upsampled.drop('label', axis = 1)     # 删除表中的某一行或者某一列
train_y = df_upsampled['label']

X = train_x
Y = train_y

X_train,X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
scaler = StandardScaler()   # 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
X_train = scaler.fit_transform(X_train)   # 计算训练集的平均值和标准差，以便测试数据集使用相同的变换
X_test = scaler.transform(X_test)

y_train = Y_train.to_numpy()
y_test = Y_test.to_numpy()

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)     # Adding a third dimension to our data
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


class ResnetBlock(Model):
  def __init__(self, filters, strides=1, residual_path=False):
    super(ResnetBlock, self).__init__()
    self.filters = filters
    self.strides = strides
    self.residual_path = residual_path

    # 第1个部分
    self.c1 = Conv1D(filters, 3, strides=strides, padding='same', use_bias=False)
    self.b1 = BatchNormalization()
    self.a1 = Activation('relu')
    # self.f1 = tf.keras.layers.Dense(1, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    # 第2个部分
    self.c2 = Conv1D(filters, 3, strides=1, padding='same', use_bias=False)
    self.b2 = BatchNormalization()
    # self.f2 = tf.keras.layers.Dense(1, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())

    # residual_path为True时，对输入进行下采样，即用1x1的卷积核做卷积操作，保证x能和F(x)维度相同，顺利相加
    if residual_path:
      self.down_c1 = Conv1D(filters, 1, strides=2, padding='same', use_bias=False)
      self.down_b1 = BatchNormalization()

    self.a2 = Activation('relu')

  def call(self, inputs):
    # residual等于输入值本身，即residual=x
    residual = inputs
    # 将输入通过卷积、BN层、激活层，计算F(x)
    x = self.c1(inputs)
    x = self.b1(x)
    x = self.a1(x)
    # x = self.f1(x)

    x = self.c2(x)
    y = self.b2(x)
    # y = self.f2(y)
    # 如果维度不同则调用代码，否则不执行
    if self.residual_path:
      residual = self.down_c1(inputs)
      residual = self.down_b1(residual)

    # 最后输出的是两部分的和，即F(x)+x或F(x)+Wx,再过激活函数
    out = self.a2(y + residual)
    return out


class ResNet18(Model):
  def __init__(self, block_list, initial_filters=64):  # block_list表示每个block有几个卷积层
    super(ResNet18, self).__init__()
    self.num_blocks = len(block_list)  # 共有几个block
    self.block_list = block_list
    self.out_filters = initial_filters
    # 结构定义
    self.c1 = Conv1D(self.out_filters, 3, strides=1, padding='same', use_bias=False)
    self.b1 = BatchNormalization()
    self.a1 = Activation('relu')
    self.blocks = tf.keras.models.Sequential()
    # 构建ResNet网络结构
    for block_id in range(len(block_list)):  # 第几个resnet block
      for layer_id in range(block_list[block_id]):  # 第几个卷积层

        if block_id != 0 and layer_id == 0:  # 对除第一个block以外的每个block的输入进行下采样
          block = ResnetBlock(self.out_filters, strides=2, residual_path=True)
        else:
          block = ResnetBlock(self.out_filters, residual_path=False)
        self.blocks.add(block)  # 将构建好的block加入resnet
      self.out_filters *= 2  # 下一个block的卷积核数是上一个block的2倍
    self.p1 = tf.keras.layers.GlobalAveragePooling1D()
    self.f1 = tf.keras.layers.Dense(10, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
    self.f2 = tf.keras.layers.Dense(1, activation='sigmoid')

  def call(self, inputs):
    x = self.c1(inputs)   # 和经典结构差了一个开头的maxpool
    x = self.b1(x)
    x = self.a1(x)
    x = self.blocks(x)
    x = self.p1(x)
    y = self.f1(x)
    y = self.f2(y)
    return y


# 运行，一共4个元素，所以block执行4次，每次有2个
model = ResNet18([2, 2, 2, 2])

# 设置优化器等
model.compile(optimizer='adam',
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

# 设置断点
checkpoint_save_path = "./checkpoint/ResNet18.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                 save_weights_only=True,
                                                 save_best_only=True)
history = model.fit(X_train, y_train, batch_size=8192, epochs=20, validation_data=(X_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])
# 显示结果
model.summary()

Y_pred = model.predict(X_test)
Y_pred = np.where(Y_pred >= 0.5, 1, 0)    # 查找符合条件的位置信息，满足≥0.5返回1，否则0
from sklearn.metrics import classification_report, recall_score, f1_score, precision_score, accuracy_score, roc_auc_score
print(classification_report(Y_test,Y_pred))
actual_acc = accuracy_score(Y_test,Y_pred)
actual_rec = recall_score(Y_test,Y_pred)
actual_p = precision_score(Y_test,Y_pred)
actual_f1 = f1_score(Y_test,Y_pred)
print('AUC ->', roc_auc_score(Y_test, Y_pred))
print('Recall ->',actual_rec)
print('Precision ->',actual_rec)
print('F1 Score ->',actual_f1)
print('Accuracy ->',actual_acc)