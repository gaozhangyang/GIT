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
            # X=df.values[:,:2]
            # Y_true = df.iloc[:,-1]
            X=df.values[:,:2][:3600]
            Y_true = df.iloc[:,-1][:3600]
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

        if name=='seismic':
            df = pd.read_csv('/home/DGC/ex2_realdata/real_data/seismic-bumps.txt',sep = ',',header=None)
            for col in [0,1,2,7]:
                sym=list(set(df.iloc[:,col]))
                for i in range(len(sym)):
                    df.iloc[df.iloc[:,col]==sym[i],col]=i
            X=df.iloc[:,:-1].values.astype(np.float)
            Y_true=df.iloc[:,-1].values.astype(np.int)
            Y_set=list(set(Y_true))
            Y_map={Y_set[i]:i for i in range(len(Y_set))}
            Y_true=np.array([Y_map[y] for y in Y_true])
            return X,Y_true

        if name=='mnist_10':
            data=np.load('/home/DGC/ex3_mnist/mnist_10.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true

        if name=='mnist_20':
            data=np.load('/home/DGC/ex3_mnist/mnist_20.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='mnist_30':
            data=np.load('/home/DGC/ex3_mnist/mnist_30.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true


def run(kd, ks, alpha,pid,process_state,idx,results):
    print('fitting model')
    Y_pred=api.DGSFC.fit(X,
                  K_d=kd,
                  K_s=ks,
                  alpha=alpha,
                  epsilon=0,
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
    save_name='mnist_20' # dataname
    pnum=10 # 并行线程数

    process_state=Manager().dict({str(i):True for i in range(pnum)})
    processes=[Process() for i in range(pnum)]
    for p in processes:
        p.start()
    
    signal.signal(signal.SIGTERM, term)

    results=Manager().dict()
    X,Y_true=DataLoader.load(save_name)
    V = len(X)
    Vmin = int(0.2*np.sqrt(V))
    Vmax = int(np.sqrt(V))

    logging.basicConfig(level=logging.INFO,#控制台打印的日志级别
                    filename='./{}.log'.format(save_name),#'log/{}_{}_{}.log'.format(args.gcn_type,args.graph_type,args.order_list)
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s: %(message)s'
                    #日志格式
                    )
    
    params = []
    # kd, ks = 45, 22
    for kd in np.arange(Vmin,Vmax,1):
        for ks in np.arange(int(0.5*kd), kd, 1):
            for alpha in np.arange(0.8,1.0,0.05):
            # for alpha in np.arange(0.89,1.00, 0.005):
                params.append((kd,ks,alpha))
    
    # params=[(50,20,0.1)]
    idx=0
    while idx<len(params):
        for pid in range(pnum):
            if process_state[str(pid)]==True:
                print(idx)
                process_state[str(pid)]=False #占用当前线程
                processes[pid].terminate()
                p=Process(target=run,args=(params[idx][0], params[idx][1],params[idx][2], pid, process_state, idx, results),name=str(pid))
                processes[pid]=p
                p.start()
                idx+=1

    for p in processes:
        p.join()
    df = pd.DataFrame.from_dict(results, orient="index")
    df.to_csv('results_{}.csv'.format(save_name))
    print('done')