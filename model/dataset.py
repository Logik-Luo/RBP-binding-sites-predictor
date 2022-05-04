import numpy as np
from configs import Bin_config
from torch.utils import data
config = Bin_config()

class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, df_feat):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_feat

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        # index = self.list_IDs[index]
        #print(self.df.iloc[index]['feat'])
        feat = eval(self.df.iloc[index]['feat'])        # 特征索引

        y = self.labels[index]

        return np.array(feat), y
