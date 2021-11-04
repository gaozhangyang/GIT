import numpy as np
import networkx as nx
from .detect_local_mode import LCluster
from .topo_graph import TopoGraph
from utils import plot_tools 


class GIT:
    def __init__(self,k,target_ratio=[1,1]):
        '''
        k: k neighbors to estimiate the local density
        target_ratio: prior class proportion
        '''
        self.k = k
        self.target_ratio = target_ratio

    @classmethod
    def fit(self,X,plot=False,draw_seed=2020):
        '''
        X: Input data
        plot: whether visualize the result or not
        draw_seed: the random seed for labeling local clusters during visualization 
        '''
        LC=LCluster()
        V,Boundary,X_extend,Dis,draw_tasks,t2=LC.detect_descending_manifolds(X,self.k,self.k)

        TG=TopoGraph()
        E_raw,E_final,final_cluster=TG.topograph_construction_pruning(V,X_extend,Boundary,Dis,self.target_ratio)

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

        else:
            return Y

# class GIT:
#     def __init__(
#         self,
#         K_d=20,
#         K_s=20,
#         epsilon=100,
#         alpha=0.0,
#         scale=False,
#     ):
#         """
#         Graph Intensity Topologg (GIT) clustering.

#         Parameters
#         ----------

#         K_d : int, default=20
#             The number of neighbors to estimate the local density.
        
#         K_s : int, default=20
#             The number of neighbors in density growing process to estimate the gradient flows. 

#         epsilon : float, default=100
#             The threshold for noise dropping.

#         alpha : float, default=0.0
#             The thresholod for topo-graph pruning.

#         scale : bool, defaut=False
#             [TO-DO]
#         """
#         self.K_d = K_d
#         self.K_s = K_s
#         self.epsilon = epsilon
#         self.alpha = alpha
#         self.scale = scale

#     def fit(self, X):
      
#         LC = LCluster(K_d=self.K_d, K_s=self.K_s, scale=self.scale)
#         V, Boundary, X_extend, Dis = LC.detect_descending_manifolds(X)

#         TG = TopoGraph(alpha=self.alpha)
#         E_raw, E = TG.topograph_construction_pruning(
#             V, X_extend, Boundary, Dis)

#         G = nx.Graph()
#         G.add_nodes_from(V.keys())
#         for e in E:
#             if E[e] > 0:
#                 G.add_edge(e[0], e[1], weight=E[e])
#         Sets = list(nx.connected_components(G))
#         if -1 in G.nodes():
#             G.remove_node(-1)

#         M2C = {}
#         for c in range(len(Sets)):
#             for m in list(Sets[c]):
#                 M2C.update({m: c})

#         for m, points in V.items():
#             X_extend[points, -1] = M2C[m]

#         for _, component in V.items():
#             if len(component) < self.epsilon:
#                 X_extend[component, -1] = -1
#         Y = X_extend[:, -1].astype(np.int)

#         return Y 
