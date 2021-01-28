import os
from multiprocessing import Process,Manager
import numpy as np
import signal
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.utils.extmath import density
# sys.path.append('/home/DGC/ex1_toy/artificial_csv')
import imp
from utils import plot_tools,api
import pandas as pd
import networkx as nx
from utils.measures import matchY,measures_calculator
from torchvision import datasets
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
import logging
from utils.kde import KDE_DIS
import pynndescent
from scipy.spatial import cKDTree
import hdbscan
from sklearn.cluster import DBSCAN


from torchvision import datasets

class DataLoader:
    def __init__(self):
        pass
    
    @classmethod
    def load(self,name):
        if name== 'circles':
            df=pd.read_csv('/home/DGC/ex1_toy/artificial_csv/circles_0.1_noise.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='moons':
            df=pd.read_csv('/home/DGC/ex1_toy/artificial_csv/moons_0.15_noise.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='impossible':
            df=pd.read_csv('/home/DGC/ex1_toy/artificial_csv/impossible_plus.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='s-set':
            df=pd.read_csv('/home/DGC/ex1_toy/artificial_csv/s-set1.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true
        
        if name=='smile':
            df=pd.read_csv('/home/DGC/ex1_toy/artificial_csv/smile1.csv', header=None)
            X=df.values[:,:2]
            Y_true = df.iloc[:,-1]
            return X,Y_true

        if name== 'iris':
            df=pd.read_csv('/home/DGC/ex2_realdata/real_data/iris.csv', header=None)
            X=df.iloc[1:,:-1].values.astype(np.float)
            Y_true=df.iloc[1:,-1].values.astype(np.float)
            return X,Y_true
        
        if name=='wine':
            df = pd.read_csv('/home/DGC/ex2_realdata/real_data/wine.csv', header=None)
            X = df.iloc[1:,:-1].values.astype(np.float)
            Y_true = df.iloc[1:,-1].values.astype(np.int)
            Y_set = list(set(Y_true))
            Y_map = {Y_set[i]:i for i in range(len(Y_set))}
            Y_true = np.array([Y_map[y] for y in Y_true])
            return X,Y_true
        
        if name=='glass':
            df = pd.read_csv('/home/DGC/ex2_realdata/real_data/glass.csv', header=0)
            X = df.iloc[:,:-1].values.astype(np.float)
            Y_true = df.iloc[:,-1].values.astype(np.int)
            Y_set = list(set(Y_true))
            Y_map = {Y_set[i]:i for i in range(len(Y_set))}
            Y_true = np.array([Y_map[y] for y in Y_true])
            return X,Y_true
        
        if name=='breast cancer':
            df = pd.read_csv('/home/DGC/ex2_realdata/real_data/wdbc.data', header=None)
            X = df.iloc[:,2:].values.astype(np.float)
            Y_true = df.iloc[:,1]

            Y_set = list(set(Y_true))
            Y_map = {Y_set[i]:i for i in range(len(Y_set))}
            Y_true = np.array([Y_map[y] for y in Y_true])
            return X,Y_true
        
        if name=='hepatitis':
            df = pd.read_csv('/home/DGC/ex2_realdata/real_data/hepatitis.data', header=None)
            df.replace('?',np.nan,inplace=True)
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.fillna(df.mean())

            X = df.iloc[1:,1:].values.astype(np.float)
            Y_true = df.iloc[1:,0].values.astype(np.int)
            Y_set = list(set(Y_true))
            Y_map = {Y_set[i]:i for i in range(len(Y_set))}
            Y_true = np.array([Y_map[y] for y in Y_true])
            return X,Y_true


def run(kd, ks, alpha,pid,process_state,idx,results):
    print('fitting model')
    Y_pred=api.DGSFC.fit(X,
                  K_d=kd,
                  K_s=ks,
                  alpha=alpha,
                  epsilon=24,
                  plot=False,
                  scale=True
                 )
    print('calculate metrics')
    result=measures_calculator(Y_true,Y_pred).values.reshape(-1)
    F,ARI,cover_rate=result[0],result[1],result[3]
   

    results[idx]=[kd, ks, alpha ,F,ARI]
    string='{:d}\t{:d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(kd, ks, alpha,F,ARI,cover_rate)
    print(string)
    logging.info(string)
    process_state[str(pid)]=True

def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        # print('the processes is {}'.format(processes) )
        for p in processes:
            # print('process {} terminate'.format(p.pid))
            p.terminate()
            # os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print(str(e))

if __name__ =='__main__':
    save_name='impossible' # dataname
   
    X,Y_true=DataLoader.load(save_name)
    kd = 41
    ks = 20
    alpha = 0.53
    idx = 0
    Y_pred=api.DGSFC.fit(X,
                K_d=kd,
                K_s=ks,
                alpha=alpha,
                epsilon=0,
                plot=True,
                scale=True
                )
    print('calculate metrics')
    result=measures_calculator(Y_true,Y_pred).values.reshape(-1)
    print(result)