import os
from multiprocessing import Process,Manager
import numpy as np
import signal
import time
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.utils.extmath import density;sys.path.append('/usr/data/gzy/code_data')
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
        if name=='mnist_5':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/mnist_5.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='mnist_10':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/mnist_10.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='mnist_20':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/mnist_20.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='mnist_30':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/mnist_30.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='fmnist_5':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/fmnist_5.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='fmnist_10':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/fmnist_10.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='fmnist_20':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/fmnist_20.npy')
            X=data[:,:-1]
            Y_true=data[:,-1]
            return X,Y_true
        
        if name=='fmnist_30':
            data=np.load('/usr/data/gzy/code_data/ex3_mnist/fmnist_30.npy')
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
    F,ARI=result[0],result[1]
   

    results[idx]=[kd, ks, alpha ,F,ARI]
    string='{:d}\t{:d}\t{:.4f}\t{:.4f}\t{:.4f}'.format(kd, ks, alpha,F,ARI)
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
    save_name='mnist_10' # dataname
    pnum=50 # 并行线程数

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
    for kd in np.arange(Vmin,Vmax,5):
        for ks in np.arange(int(0.5*kd), kd, 5):
            for alpha in np.arange(0.1,0.5,0.1):
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