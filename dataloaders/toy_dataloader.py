import pandas as pd
import os.path as osp


toy_datasets = ['circles', 'impossible', 'moons', 's-set', \
                'smile', 'complex8', 'complex9', 'chainlink']

class Toy_DataLoader:
    def __init__(self, name, path='./datasets/toy_datasets'):
        self.name = name
        self.path = path
        assert name in toy_datasets

    def load(self):
        df = pd.read_csv(osp.join(self.path, self.name + '.csv'), header=None)
        if self.name == 'chainlink':
            X, Y_true = df.values[:,:3], df.iloc[:,-1]
        else:
            X, Y_true = df.values[:, :2], df.iloc[:, -1]
        return X, Y_true