import numpy as np
from .detect_local_mode import LCluster
from .topo_graph import TopoGraph


class GIT:
    def __init__(
        self, 
        k=8, 
        target_ratio=[1., 1.],
        n_jobs=100
    ):
        """
        Graph Intensity Topologg (GIT) clustering.
        Parameters
        ----------
        k : int, default=8
            The number of neighbors to estimate the local density.
        target_ratio : list of float, default=[1., 1.]
            [TODO]
        n_jobs : int, default=100
            The number of threads used in `NearestNeighbors`.
        """
        self.k = k
        self.target_ratio = target_ratio
        self.n_jobs = n_jobs

    def fit_predict(self, X):
        """Compute GIT clustering.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
        """
        LC = LCluster(k=self.k, n_jobs=self.n_jobs)
        V, Boundary, X_extend, Dis = LC.detect_descending_manifolds(X)

        TG = TopoGraph(target_ratio=self.target_ratio)
        _, _, clusters = TG.topograph_construction_pruning(
            V, X_extend, Boundary, Dis)

        for c, cluster_index in clusters.items():
            for cluster in cluster_index:
                X_extend[V[cluster], -1] = c

        Y = X_extend[:, -1].astype(np.int)
        return Y