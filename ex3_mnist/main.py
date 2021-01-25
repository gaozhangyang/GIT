import sys;sys.path.append('/usr/data/gzy/rebuttal_DGC/DGC')
from sklearn.utils import shuffle
from utils import kde
import numpy as np
import matplotlib.pyplot as plt
from utils import plot_tools
import pandas as pd
import networkx as nx
from utils.measures import matchY,measures_calculator
from utils import api
from torchvision import datasets

dataset2 = datasets.MNIST('../data', train=False, download=True)


class DataLoader:
    def __init__(self):
        pass
    
    @classmethod
    def load(self,name):
        if name== 'circles':
            df=pd.read_csv('./artificial_csv/circles_0.1_noise.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='moons':
            df=pd.read_csv('./artificial_csv/moons_0.15_noise.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='impossible':
            df=pd.read_csv('./artificial_csv/impossible_plus.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='s-set':
            df=pd.read_csv('./artificial_csv/s-set1.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='smile':
            df=pd.read_csv('./artificial_csv/smile1.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='mnist':
            X=dataset2.test_data.numpy().reshape(10000,784)
            Y_true=dataset2.test_labels.numpy()
            return X,Y_true


if __name__ =='__main__':
    X,Y_true=DataLoader.load('mnist')
    # shuffle_idx=np.arange(X.shape[0])
    # np.random.shuffle(shuffle_idx)
    # shuffle_idx=np.sort(Y_true)
    # X,Y_true=X[shuffle_idx],Y_true[shuffle_idx]
    scales=kde.KDE_DIS.get_scales(X)
    useful_dim=(np.array(scales)>0)
    X=X[:,useful_dim]

    Y_pred=api.DGSFC.fit(  X,
                    K_d=100,
                    K_s=100,
                    alpha=0.8,
                    epsilon=0,
                    plot=False,
                    )
    # plot_tools.autoPlot(X,Y_pred)

    Y_pred,Y_true=matchY(Y_pred,Y_true)

    result=measures_calculator(Y_true, Y_pred)
    print(result)