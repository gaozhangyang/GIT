import numpy as np
norm_minmax = lambda x:(x-np.min(x)) / (np.max(x)-np.min(x))

from sklearn.neighbors import NearestNeighbors


class KDE_DIS():
    def __init__(
        self,
        dataset,
        K,
        n_jobs=100
    ):
        self.dataset=dataset
        scales = self.get_scales(self.dataset)
        self.index = np.where(np.array(scales) != 0.0)[0]
        self.A=1 / (np.array(scales)[self.index])
        self.K = K 
        self.dataset_ = np.matmul(self.dataset[:, self.index],np.diag(self.A))
        self.NN = NearestNeighbors(n_neighbors=self.K, metric='minkowski', p=2,metric_params=None, n_jobs=n_jobs).fit(self.dataset_)
        self.G = self.NN.kneighbors_graph(X=None, n_neighbors=self.K, mode='distance')
        self.D = self.G.data.reshape(-1,self.K)
        self.I = self.G.indices.reshape(-1,self.K)

        N = dataset.shape[0]
        self.D = np.hstack([np.zeros(N).reshape(-1,1),self.D])
        self.I = np.hstack([np.arange(N).reshape(-1,1),self.I])
        name = 'local_density'
        if name=='knn_density':
            self.P = 1/self.D[:,-1]**self.dataset.shape[1]

        if name=='local_density':
            D = self.D/np.sqrt(dataset.shape[1])
            EXP = np.exp(-D[:,1:]**2)
            self.P = np.mean(EXP,axis=1)
    
    def get_scales(self,X_observed):
        X_ = X_observed[:,:]
        Sigma_ = []
        Scale_ = []

        for i in range(X_.shape[1]):
            sigma=np.sqrt(np.var(X_[:,i]))
            scale=sigma
            Sigma_.append(sigma)
            Scale_.append(scale)
        return Scale_

    def get_DI(self,idx):
        idx = idx.astype(np.int)
        return self.D[idx],self.I[idx]
    
    def get_midP(self,idxL,idxR):
        P_mid = (self.P[idxL]+self.P[idxR])/2
        return P_mid
    
    def get_density(self,idx):
        idx = idx.astype(np.int)
        return self.P[idx]