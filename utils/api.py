from random import seed
from networkx.readwrite.json_graph import tree
import numpy as np,random
import networkx as nx
import sys;sys.path.append('/home/DGC/utils')
from detect_local_mode import LCluster
from topo_graph import TopoGraph
import plot_tools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import pickle
import time



class GIT:
    def __init__(self) -> None:
        pass

    @classmethod
    def fit(self,X,
                 k,
                 target_ratio=[1,1],
                 plot=False,
                 draw_seed=2020
                 ):
        '''
        K_d: k neighbors to estimiate the local density
        K_n: K neighbors in density growing process, for estimate the gradient flows
        epsilon: the threshold for noise dropping
        alpha: the threshold for topo-graph pruning
        '''
        t1=time.time()
        LC=LCluster()
        V,Boundary,X_extend,Dis,draw_tasks,t2=LC.detect_descending_manifolds(X,k,k)
        t3=time.time()

        TG=TopoGraph()
        E_raw,E_final,final_cluster=TG.topograph_construction_pruning(V,X_extend,Boundary,Dis,target_ratio)
        t4=time.time()

        # node2cls: index of local cluster --> index of classes
        # c2ordered: index of classes --> index of ordered classes, Y=0,1,2,...
        # c2ordered = { c:i for i,c in enumerate(set(node2cls.values()))}
        for c,cluster_index in final_cluster.items():
            for cluster in cluster_index:
                X_extend[V[cluster],-1]=c

        Y = X_extend[:,-1].astype(np.int)

        ########################plot######################
        if plot:
            Dim=X_extend.shape[1]
            if Dim<=6:
                # show raw data
                plot_tools.autoPlot(X,np.zeros(X_extend.shape[0]).astype(np.int))
                # show density
                plot_tools.autoPlot(X, y=X_extend[:, -4],continues=True)
                # show local modes
                plot_tools.PaperGraph.show_local_clusters(X,V,seed=draw_seed)
                # show topo-graph
                plot_tools.PaperGraph.show_topo_graph(V,E_raw,X)
                # show pruned topo-graph
                plot_tools.PaperGraph.show_topo_graph(V,E_final,X)
                # show clustering results
                plot_tools.autoPlot(X,Y)
            else:
                # show pruned topo-graph
                plot_tools.PaperGraph.show_topo_graph(V,E_final)
        return Y,V,E_raw,X_extend,draw_tasks,final_cluster#,t1,t2,t3,t4,t5


if __name__ =='__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import sys
    import plot_tools,api
    import pandas as pd
    import networkx as nx


    # class DataLoader:
    #     def __init__(self):
    #         pass
        
    #     @classmethod
    #     def load(self,name):
    #         if name== 'iris':
    #             df=pd.read_csv('./real_data/iris.csv', header=None)
    #             X=df.iloc[1:,:-1].values.astype(np.float)
    #             Y_true=df.iloc[1:,-1].values.astype(np.float)
    #             return X,Y_true
            
    #         if name=='wine':
    #             df = pd.read_csv('./real_data/wine.csv', header=None)
    #             X = df.iloc[1:,:-1].values.astype(np.float)
    #             Y_true = df.iloc[1:,-1].values.astype(np.int)
    #             Y_set = list(set(Y_true))
    #             Y_map = {Y_set[i]:i for i in range(len(Y_set))}
    #             Y_true = np.array([Y_map[y] for y in Y_true])
    #             return X,Y_true
            
    #         if name=='glass':
    #             df = pd.read_csv('/usr/data/gzy/rebuttal_DGC/DGC_gzy/DGC/ex3_realdata/real_data/glass.csv', header=0)
    #             X = df.iloc[:,:-1].values.astype(np.float)
    #             Y_true = df.iloc[:,-1].values.astype(np.int)
    #             Y_set = list(set(Y_true))
    #             Y_map = {Y_set[i]:i for i in range(len(Y_set))}
    #             Y_true = np.array([Y_map[y] for y in Y_true])
    #             return X,Y_true
            
    #         if name=='breast cancer':
    #             df = pd.read_csv('./real_data/wdbc.data', header=None)
    #             X = df.iloc[:,2:].values.astype(np.float)
    #             Y_true = df.iloc[:,1]

    #             Y_set = list(set(Y_true))
    #             Y_map = {Y_set[i]:i for i in range(len(Y_set))}
    #             Y_true = np.array([Y_map[y] for y in Y_true])
    #             return X,Y_true
            
    #         if name=='hepatitis':
    #             df = pd.read_csv('./real_data/hepatitis.data', header=None)
    #             df.replace('?',np.nan,inplace=True)
    #             df = df.apply(pd.to_numeric, errors='coerce')
    #             df = df.fillna(df.mean())

    #             X = df.iloc[1:,1:].values.astype(np.float)
    #             Y_true = df.iloc[1:,0].values.astype(np.int)
    #             Y_set = list(set(Y_true))
    #             Y_map = {Y_set[i]:i for i in range(len(Y_set))}
    #             Y_true = np.array([Y_map[y] for y in Y_true])
    #             return X,Y_true


    class DataLoader:
        def __init__(self):
            pass
        
        @classmethod
        def load(self,name):
            if name== 'circles':
                df=pd.read_csv('/usr/data/gzy/code_data/ex1_toy/artificial_csv/circles_0.1_noise.csv', header=None)
                X=df.values[:,:2]
                Y_true = df.iloc[:,-1]
                return X,Y_true
            
            if name=='moons':
                df=pd.read_csv('/usr/data/gzy/code_data/ex1_toy/artificial_csv/moons_0.15_noise.csv', header=None)
                X=df.values[:,:2]
                Y_true = df.iloc[:,-1]
                return X,Y_true
            
            if name=='impossible':
                df=pd.read_csv('/usr/data/gzy/code_data/ex1_toy/artificial_csv/impossible_plus.csv', header=None)
                X=df.values[:,:2]
                Y_true = df.iloc[:,-1]
                return X,Y_true
            
            if name=='s-set':
                df=pd.read_csv('/usr/data/gzy/code_data/ex1_toy/artificial_csv/s-set1.csv', header=None)
                X=df.values[:,:2]
                Y_true = df.iloc[:,-1]
                return X,Y_true
            
            if name=='smile':
                df=pd.read_csv('/usr/data/gzy/code_data/ex1_toy/artificial_csv/smile1.csv', header=None)
                X=df.values[:,:2]
                Y_true = df.iloc[:,-1]
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



    X,Y_true=DataLoader.load('impossible')

    Y_pred,V,E_raw,X_extend,draw_tasks,final_cluster=api.GIT.fit(  X,
                    k=8,
                    target_ratio=[2, 2, 1, 1, 1, 1, 1],
                    plot=False,
                    )
    # plot_tools.autoPlot(X,Y_pred)