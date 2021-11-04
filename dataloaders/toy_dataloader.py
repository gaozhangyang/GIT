import pandas as pd
import os.path as osp


toy_datasets = ['circles', 'impossible', 'moons', 's-set', 'smile']

class Toy_DataLoader:
    def __init__(self, name, path='./datasets/toy_datasets'):
        self.name = name
        self.path = path
        assert name in toy_datasets

    def load(self):
        df = pd.read_csv(osp.join(self.path, self.name + '.csv'), header=None)
        X = df.values[:, :2]
        Y_true = df.iloc[:, -1]
        return X, Y_true