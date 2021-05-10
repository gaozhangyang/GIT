import scipy
import numpy as np
from scipy.spatial import cKDTree
norm_minmax=lambda x:(x-np.min(x))/(np.max(x)-np.min(x))

# class KDE_DIS():
#     def __init__(self,dataset,param,scale):
#         self.dataset=dataset
#         scales = self.get_scales(self.dataset)
#         self.index = np.where(np.array(scales) != 0.0)[0]
#         self.A=1 / (np.array(scales)[self.index])
#         self.param = (param+1).item() if isinstance(param, np.int64) else param+1
#         dataset_=np.matmul(self.dataset[:, self.index],np.diag(self.A))
#         self.kdtree=cKDTree(dataset_)
#         self.scale=scale
    
#     def get_scales(self,X_observed):
#         X_ = X_observed[:,:]
#         Sigma_ = []
#         Scale_ = []

#         for i in range(X_.shape[1]):
#             sigma=np.sqrt(np.var(X_[:,i]))
#             # scale=(4*sigma**5/(3*X_[:,i].shape[0]))**0.2
#             scale=sigma
#             Sigma_.append(sigma)
#             Scale_.append(scale)
#         return Scale_
    
#     def get_DI(self,X):
#         X_=np.matmul(X[:, self.index],np.diag(self.A))
#         self.D, self.I = self.kdtree.query(X_, k=self.param)
#         return self.D,self.I

#     def knn_graph(self):
#         D,I=self.get_DI(self.dataset,self.param,'knn')
#         W=np.zeros((D.shape[0],D.shape[0]))
#         for i in range(0,D.shape[0]):
#             W[i,I[i]]=D[i]
#         return W
    
#     def get_density(self,X,train=False):
#         D, I=self.get_DI(X)

#         if self.scale:
#             D=(D/np.sqrt(X.shape[1]))
#             EXP=np.exp(-D[:,1:]**2)
#         else:
#             EXP=np.exp(-D**2)
#         P=np.mean(EXP,axis=1)
#         # if train:
#         #     self.maxP=np.max(P)
#         #     self.minP=np.min(P)
#         # P=(P-self.minP)/(self.maxP-self.minP+1e-10)
#         return P,D,I

from sklearn.neighbors import NearestNeighbors


class KDE_DIS():
    def __init__(self,dataset,K,scale,n_jobs=100):
        self.dataset=dataset
        scales = self.get_scales(self.dataset)
        self.index = np.where(np.array(scales) != 0.0)[0]
        self.A=1 / (np.array(scales)[self.index])
        self.K = K #(K+1).item() if isinstance(K, np.int64) else K+1
        self.dataset_=np.matmul(self.dataset[:, self.index],np.diag(self.A))
        self.NN = NearestNeighbors(n_neighbors=self.K, metric='minkowski', p=2,metric_params=None, n_jobs=n_jobs).fit(self.dataset_)
        self.G = self.NN.kneighbors_graph(X=None, n_neighbors=self.K, mode='distance')
        self.D = self.G.data.reshape(-1,self.K)
        self.I = self.G.indices.reshape(-1,self.K)

        N = dataset.shape[0]
        self.D = np.hstack([np.zeros(N).reshape(-1,1),self.D])
        self.I = np.hstack([np.arange(N).reshape(-1,1),self.I])
        D = self.D/np.sqrt(dataset.shape[1])
        EXP = np.exp(-D[:,1:]**2)
        self.P = np.mean(EXP,axis=1)
        self.scale=scale
    
    def get_scales(self,X_observed):
        X_ = X_observed[:,:]
        Sigma_ = []
        Scale_ = []

        for i in range(X_.shape[1]):
            sigma=np.sqrt(np.var(X_[:,i]))
            # scale=(4*sigma**5/(3*X_[:,i].shape[0]))**0.2
            scale=sigma
            Sigma_.append(sigma)
            Scale_.append(scale)
        return Scale_

    def get_DI(self,idx):
        idx = idx.astype(np.int)
        return self.D[idx],self.I[idx]
    
    def get_midP(self,idxL,idxR):
        # X_left = self.dataset_[idxL]
        # X_right = self.dataset_[idxR]
        # X_mid = (X_left+X_right)/2
        # neigh_dist, neigh_ind= self.NN.kneighbors(X_mid,self.K,return_distance=True)

        # neigh_dist = neigh_dist/np.sqrt(self.dataset_.shape[1])
        # EXP = np.exp(-neigh_dist**2)
        # P_mid = np.mean(EXP,axis=1)

        P_mid = (self.P[idxL]+self.P[idxR])/2
        return P_mid
    
    def get_density(self,idx):
        idx = idx.astype(np.int)
        return self.P[idx]