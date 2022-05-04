from dataset import *
import numpy as np
from dataset import *
from configs import Bin_config
from torch.utils import data
from tqdm import tqdm
import torch
import copy
from models import *
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
import pandas as pd
import tensorflow as tf
# from imblearn.over_sampling import ADASYN
# from sklearn.svm import SVC


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def test(data_generator, model):
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (feat, label) in enumerate(tqdm(data_generator)):

        score = model(feat.to(device).float())

        logits = torch.squeeze(score)
        loss_fct = torch.nn.BCELoss()

        label = Variable(torch.from_numpy(np.array(label)).to(device).float())

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()


    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision = tpr / (tpr + fpr + 0.00001)

    f1 = 2 * precision * tpr / (tpr + precision + 0.00001)
    thred_optim = thresholds[1:][np.argmax(f1[1:])]
    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]
    cm1 = confusion_matrix(y_label, y_pred_s)
    total1 = sum(sum(cm1))
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1        # acc求的正阳，负阴都对的概率

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    return roc_auc_score(y_label, y_pred, average="micro"), average_precision_score(y_label, y_pred, average="micro"), f1_score(y_label,outputs, average="micro"), \
           accuracy1, recall_score(y_label, y_pred_s), precision_score(y_label, y_pred_s, average="micro")


def main():
    config = Bin_config()

    if config['model'] == 'cnn':
        model = CNN(**config).to(device)
    elif config['model'] == 'lstm':
        model = LSTM(**config).to(device)
    elif config['model'] == 'gru':
        model = GRU(**config).to(device)
    elif config['model'] == 'transformer':
        model = Transformer(**config).to(device)
    elif config['model'] == 'cnn_maxpool':
        model = CNN_MaxPool(**config).to(device)

    params = {'batch_size': config['batch_size'],
              'shuffle': False,
              'num_workers': config['num_workers'],
              'drop_last': True}

    df_train = pd.read_csv('data/train.csv')
    training_set = Dataset(df_train.index.values, df_train.label.values, df_train)
    training_generator = data.DataLoader(training_set, **params)


    df_test = pd.read_csv('data/test.csv')
    testing_set = Dataset(df_test.index.values, df_test.label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    df_val = pd.read_csv('data/val.csv')
    validation_set = Dataset(df_val.index.values, df_val.label.values, df_val)
    validation_generator = data.DataLoader(validation_set, **params)

    max_auc = 0
    model_max = copy.deepcopy(model)

    opt = torch.optim.Adamax(model.parameters(), lr=config['lr'])                     # 改优化器！

    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True       # 设置这个 flag 可以让内置的 cuDNN 的 auto-tuner 自动寻找最适合当前配置的高效算法
    for epo in range(config['epochs']):
        model.train()
        for i, (feat, label) in enumerate(tqdm(training_generator)):
            score = model(feat.to(device).float())

            label = Variable(torch.from_numpy(np.array(label)).float()).to(device)
            loss_fct = torch.nn.BCELoss()
            m = torch.squeeze(score)
            loss = loss_fct(m, label)

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i % 1000 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        with torch.set_grad_enabled(False):
            auc, auprc, f1, acc, rec, pre = test(validation_generator, model)
            print("[Validation metrics]: auc:{:.4f}, auprc:{:.4f}, f1:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, precision:{:.4f}".format(auc, auprc, f1, acc, rec, pre))

            if auc > max_auc:
                model_max = copy.deepcopy(model)
                torch.save(model_max, 'save_model/model.pth')
                max_auc = auc
                print("*" * 30 + " save best model " + "*" * 30)

        torch.cuda.empty_cache()

    print('\n--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc, auprc, f1, acc, rec, pre = test(testing_generator, model)
            print("[Testing metrics]: auc:{:.4f}, auprc:{:.4f}, f1:{:.4f}, accuracy:{:.4f}, recall:{:.4f}, precision:{:.4f}".format(auc, auprc, f1, acc, rec, pre))
    except:
        print('testing failed')


if __name__ == '__main__':

    main()
    auc_list = []
    for epoch in range(1, config['epochs'] + 1):
        # Train cycle
        trainModel()
        auc = testModel()
        auc_list.append(auc)  # 存入列表，后面画图使用

    # 画图
    epoch = np.arange(1, len(auc_list) + 1, 1)  # 步长为1
    auc_list = np.array(auc_list)
    plt.plot(epoch, auc_list)
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.grid()  # 显示网格线 1=True=默认显示；0=False=不显示
    plt.show()
