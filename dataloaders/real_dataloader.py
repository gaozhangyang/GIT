import numpy as np
import pandas as pd
import os.path as osp

real_datasets = ['iris', 'wine', 'glass', 'breast_cancer', 'hepatitis']

class Real_DataLoader:
    def __init__(self, name, path='../datasets/real_datasets/'):
        self.name = name
        self.path = path
        assert name in real_datasets

    def load(self):
        df = pd.read_csv(osp.join(self.path, self.name + '.csv'), header=None)
        if self.name == 'iris':
            X = df.iloc[1:, :-1].values.astype(np.float)
            Y_true = df.iloc[1:, -1].values.astype(np.float)
        elif self.name == 'wine':
            X = df.iloc[1:, :-1].values.astype(np.float)
            Y_true = df.iloc[1:, -1].values.astype(np.int)
            Y_set = list(set(Y_true))
            Y_map = {Y_set[i]: i for i in range(len(Y_set))}
            Y_true = np.array([Y_map[y] for y in Y_true])
        elif self.name == 'breast_cancer':
            X = df.iloc[:, 2:].values.astype(np.float)
            Y_true = df.iloc[:, 1]
            Y_set = list(set(Y_true))
            Y_map = {Y_set[i]: i for i in range(len(Y_set))}
            Y_true = np.array([Y_map[y] for y in Y_true])
        elif self.name == 'hepatitis':
            df.replace('?', np.nan, inplace=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(df.mean())
            X = df.iloc[1:, 1:].values.astype(np.float)
            Y_true = df.iloc[1:, 0].values.astype(np.int)
            Y_set = list(set(Y_true))
            Y_map = {Y_set[i]: i for i in range(len(Y_set))}
            Y_true = np.array([Y_map[y] for y in Y_true])
        return X, Y_true
