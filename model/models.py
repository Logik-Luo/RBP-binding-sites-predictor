import torch
from torch import nn
from configs import Bin_config

config = Bin_config()


class CNN(nn.Sequential):
    def __init__(self, **config):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(38,64,1)

        self.conv2 = nn.Conv1d(64,64,1)

        self.decoder = nn.Sequential(
            nn.Linear(config['flatten_dim'], 4),
            nn.ReLU(True),

            nn.BatchNorm1d(4),
            nn.Linear(4, 1),

            nn.Sigmoid()
        )

    def forward(self, feat):
        feat = feat.view(config['batch_size'], 38, 1)
        # print("init_feat: ", feat.shape)

        feat = self.conv1(feat)
        # print("conv_feat: ", feat.shape)
        feat = self.conv2(feat)

        feat = feat.view(config['batch_size'], -1)
        # print("view_feat: ", feat.shape)
        feat = self.decoder(feat)
        # print("result_feat: ", feat.shape)
        return feat

class CNN_MaxPool(nn.Sequential):
    def __init__(self, **config):
        super(CNN_MaxPool, self).__init__()

        self.conv1 = nn.Conv1d(38,24,1)
        self.maxpool1 = nn.MaxPool1d(1, stride=1)
        self.conv2 = nn.Conv1d(24,12,1)
        self.maxpool2 = nn.MaxPool1d(1, stride=1)
        self.decoder = nn.Sequential(
            nn.Linear(config['flatten_dim'], 4),
            nn.ReLU(True),

            nn.BatchNorm1d(4),
            nn.Linear(4, 1),

            nn.Sigmoid()
        )

    def forward(self, feat):
        feat = feat.view(config['batch_size'], 38, 1)
        feat = self.conv1(feat)
        feat = self.maxpool1(feat)
        feat = self.conv2(feat)
        feat = self.maxpool2(feat)
        feat = feat.view(config['batch_size'], -1)
        # print("view_feat: ", feat.shape)
        feat = self.decoder(feat)
        # print("result_feat: ", feat.shape)
        return feat

class LSTM(nn.Sequential):
    def __init__(self, **config):
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=1, hidden_size=16, num_layers=2, bidirectional=True)

        self.decoder = nn.Sequential(
            nn.Linear(config['flatten_dim'], 4),
            nn.ReLU(True),

            nn.BatchNorm1d(4),
            nn.Linear(4, 1),
          
            nn.Sigmoid()

        )

    def forward(self, feat):
        feat = feat.view(config['batch_size'], 38, 1)
        # print("feat: ", feat.shape)

        feat, _ = self.rnn(feat)

        # print("lstm_feat: ", feat.shape)

        feat = feat.view(config['batch_size'], -1)
        #print("view_feat: ", feat.shape)
        feat = self.decoder(feat)
    
        return feat

class GRU(nn.Sequential):
    def __init__(self, **config):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(input_size=1, hidden_size=16, num_layers=2, bidirectional=True)

        self.decoder = nn.Sequential(
            nn.Linear(config['flatten_dim'], 4),
            nn.ReLU(True),

            nn.BatchNorm1d(4),
            nn.Linear(4, 1),
          
            nn.Sigmoid()

        )

    def forward(self, feat):
        feat = feat.view(config['batch_size'], 38, 1)
        # print("feat: ", feat.shape)

        feat, _ = self.rnn(feat)

        feat = feat.view(config['batch_size'], -1)
        # print("view_feat: ", feat.shape)
        feat = self.decoder(feat)
    
        return feat

class Transformer(nn.Sequential):
    def __init__(self, **config):
        super(Transformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=1, nhead=1, dim_feedforward=16, dropout=0.1, activation='relu')

        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.decoder = nn.Sequential(
            nn.Linear(config['flatten_dim'], 4),
            nn.ReLU(True),

            nn.BatchNorm1d(4),
            nn.Linear(4, 1),

            nn.Sigmoid()

        )

    def forward(self, feat):
        feat = feat.view(config['batch_size'], 38, 1)
        # print("feat: ", feat.shape)

        feat = self.trans(feat)

        feat = feat.view(config['batch_size'], -1)
        #print("view_feat: ", feat.shape)
        feat = self.decoder(feat)

        return feat