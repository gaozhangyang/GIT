from random import seed
from networkx.readwrite.json_graph import tree
import numpy as np,random
import networkx as nx
import sys;sys.path.append('/usr/data/gzy/code_data/utils')
from detect_local_mode import LCluster
from topo_graph import TopoGraph
import plot_tools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import pickle

class DGSFC:
    def __init__(self) -> None:
        pass

    @classmethod
    def fit(self,X,
                 K_d,
                 K_s,
                 epsilon=1,
                 alpha=0,
                 plot=False,
                 scale=False,
                 draw_seed=2020
                 ):
        '''
        K_d: k neighbors to estimiate the local density
        K_n: K neighbors in density growing process, for estimate the gradient flows
        epsilon: the threshold for noise dropping
        alpha: the threshold for topo-graph pruning
        '''
        LC=LCluster()
        V,Boundary,X_extend,Dis,draw_tasks=LC.detect_descending_manifolds(X,K_d,K_s,epsilon,scale)

        TG=TopoGraph()
        E_raw,E=TG.topograph_construction_pruning(X_extend,Boundary,Dis,alpha)

        G = nx.Graph()
        G.add_nodes_from(V.keys())
        for e in E:
            if E[e]>0:
                G.add_edge(e[0],e[1],weight=E[e])
        Sets=list(nx.connected_components(G))
        if -1 in G.nodes():
            G.remove_node(-1)

        M2C={}
        for c in range(len(Sets)):
            for m in list(Sets[c]):
                M2C.update({m:c})

        for m,points in V.items():
            X_extend[points,-1]=M2C[m]
        
        if -1 in V.keys():
            X_extend[V[-1],-1]=-1
        Y=X_extend[:,-1].astype(np.int)


        ########################plot######################
        if plot:
            Dim=X_extend.shape[1]
            if Dim<=6:
                # show raw data
                plot_tools.autoPlot(X,np.zeros(X_extend.shape[0]).astype(np.int))
                # show local modes
                plot_tools.PaperGraph.show_local_clusters(X,V,seed=draw_seed)
                # show topo-graph
                plot_tools.PaperGraph.show_topo_graph(V,E_raw,X)
                # show pruned topo-graph
                plot_tools.PaperGraph.show_topo_graph(V,E,X)
                # show clustering results
                plot_tools.autoPlot(X,Y)
            else:
                # show pruned topo-graph
                plot_tools.PaperGraph.show_topo_graph(V,E)
        return Y