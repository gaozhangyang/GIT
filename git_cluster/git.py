import numpy as np
import networkx as nx
from .detect_local_mode import LCluster
from .topo_graph import TopoGraph


class GIT:
    def __init__(
        self,
        K_d=20,
        K_s=20,
        epsilon=100,
        alpha=0.0,
        scale=False,
    ):
        """
        Graph Intensity Topologg (GIT) clustering.

        Parameters
        ----------

        K_d : int, default=20
            The number of neighbors to estimate the local density.
        
        K_s : int, default=20
            The number of neighbors in density growing process to estimate the gradient flows. 

        epsilon : float, default=100
            The threshold for noise dropping.

        alpha : float, default=0.0
            The thresholod for topo-graph pruning.

        scale : bool, defaut=False
            [TO-DO]
        """
        self.K_d = K_d
        self.K_s = K_s
        self.epsilon = epsilon
        self.alpha = alpha
        self.scale = scale

    def fit(self, X):
      
        LC = LCluster(K_d=self.K_d, K_s=self.K_s, scale=self.scale)
        V, Boundary, X_extend, Dis = LC.detect_descending_manifolds(X)

        TG = TopoGraph(alpha=self.alpha)
        E_raw, E = TG.topograph_construction_pruning(
            V, X_extend, Boundary, Dis)

        G = nx.Graph()
        G.add_nodes_from(V.keys())
        for e in E:
            if E[e] > 0:
                G.add_edge(e[0], e[1], weight=E[e])
        Sets = list(nx.connected_components(G))
        if -1 in G.nodes():
            G.remove_node(-1)

        M2C = {}
        for c in range(len(Sets)):
            for m in list(Sets[c]):
                M2C.update({m: c})

        for m, points in V.items():
            X_extend[points, -1] = M2C[m]

        for _, component in V.items():
            if len(component) < self.epsilon:
                X_extend[component, -1] = -1
        Y = X_extend[:, -1].astype(np.int)

        return Y 
