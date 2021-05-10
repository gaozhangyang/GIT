import scipy
import numpy as np
from scipy.spatial import cKDTree
norm_minmax=lambda x:(x-np.min(x))/(np.max(x)-np.min(x))

class KDE_DIS():
    def __init__(self,dataset,param,scale):
        self.dataset=dataset
        scales = self.get_scales(self.dataset)
        self.index = np.where(np.array(scales) != 0.0)[0]
        self.A=1 / (np.array(scales)[self.index])
        self.param = (param+1).item() if isinstance(param, np.int64) else param+1
        dataset_=np.matmul(self.dataset[:, self.index],np.diag(self.A))
        self.kdtree=cKDTree(dataset_)
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
    
    def get_DI(self,X):
        X_=np.matmul(X[:, self.index],np.diag(self.A))
        self.D, self.I = self.kdtree.query(X_, k=self.param)
        return self.D,self.I

    def knn_graph(self):
        D,I=self.get_DI(self.dataset,self.param,'knn')
        W=np.zeros((D.shape[0],D.shape[0]))
        for i in range(0,D.shape[0]):
            W[i,I[i]]=D[i]
        return W
    
    def get_density(self,X,train=False):
        D, I=self.get_DI(X)

        if self.scale:
            D=(D/np.sqrt(X.shape[1]))
            EXP=np.exp(-D[:,1:]**2)
        else:
            EXP=np.exp(-D**2)
        P=np.mean(EXP,axis=1)
        # if train:
        #     self.maxP=np.max(P)
        #     self.minP=np.min(P)
        # P=(P-self.minP)/(self.maxP-self.minP+1e-10)
        return P,D,I