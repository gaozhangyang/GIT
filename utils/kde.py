import scipy
import numpy as np

norm_minmax=lambda x:(x-np.min(x))/(np.max(x)-np.min(x))

class KDE_DIS():
    def __init__(self,dataset,param):
        self.dataset=dataset
        self.A=1/ np.array(self.get_scales(self.dataset))
        self.param=param+1
    
    def get_scales(self,X_observed):
        X_ = X_observed[:,:]
        Sigma_ = []
        Scale_ = []

        for i in range(X_.shape[1]):
            sigma=np.sqrt(np.var(X_[:,i]))
            scale=(4*sigma**5/(3*X_[:,i].shape[0]))**0.2
            Sigma_.append(sigma)
            Scale_.append(scale)
        return Scale_
    
    def get_DI(self,X,param):
        A=np.diag(self.A)
        D_mat=scipy.spatial.distance_matrix(np.matmul(X,A),np.matmul(self.dataset,A))
        index=np.argsort(D_mat,axis=1)
        I=index[:,:param]
        D=np.zeros_like(I,dtype=np.float)
        for row in range(D.shape[0]):
            D[row]=D_mat[row,I[row]]
        return D,I
            
    
    def knn_graph(self):
        D,I=self.get_DI(self.dataset,self.param+1,'knn')
        W=np.zeros((D.shape[0],D.shape[0]))
        for i in range(0,D.shape[0]):
            W[i,I[i]]=D[i]
        return W
    
    def get_density(self,X,k,train=False):
        D, I=self.get_DI(X,param=k)

        EXP=np.exp(-D**2)
        P=np.mean(EXP,axis=1)
        if train:
            self.maxP=np.max(P)
            self.minP=np.min(P)
        P=(P-self.minP)/(self.maxP-self.minP+1e-10)
        return P,D,I