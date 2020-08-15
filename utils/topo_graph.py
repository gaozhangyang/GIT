import numpy as np

class TopoGraph:
    def __init__(self):
        pass
    
    #####################################################################
    #######################connectivity of local clusters################
    @classmethod
    def get_boundary(self,connection,manifolds):
        connection=np.array(connection)
        BoundaryMat_E=np.zeros((len(manifolds),len(manifolds)),dtype=set)
        for i in range(len(manifolds)):
            for j in range(i+1,len(manifolds)):
                maski2j = (connection[:,2]==manifolds[i].rt) & (connection[:,3]==manifolds[j].rt)
                maskj2i = (connection[:,2]==manifolds[j].rt) & (connection[:,3]==manifolds[i].rt)
                BoundaryMat_E[i,j] = BoundaryMat_E[j,i] =connection[maski2j|maskj2i,:2]
        return BoundaryMat_E
    
    @classmethod
    def mean_center(self,X_extend,idx):
        left,right=idx[:,0],idx[:,1]
        new_pos=[]
        for dim in range(X_extend.shape[1]-2):
            new_pos.append(((X_extend[left,dim]+X_extend[right,dim])/2).reshape(-1,1))
        return np.hstack(new_pos)

    @classmethod
    def connectivity_of_2_clusters(self,i,j,K_d,Dis,BoundaryMat_E,X_extend,manifolds):
        BE=BoundaryMat_E[i,j]
        if BE.shape[0]==0:
            return 0
        P_mid,D,I=Dis.get_density(TopoGraph.mean_center(X_extend,BE),K_d+1,train=True)

        P1=X_extend[manifolds[i].rt,-2]
        P2=X_extend[manifolds[j].rt,-2]
        return np.sum(P_mid**2)*min(P1/P2,P2/P1)**2
    
    @classmethod
    def connectivity_all(self,real_manifolds,K_d,Dis,BoundaryMat_E,X_extend):
        ConnectMat=np.zeros([len(real_manifolds),len(real_manifolds)])
        for i in range(len(real_manifolds)):
            for j in range(i+1,len(real_manifolds)):
                ConnectMat[i,j]=ConnectMat[j,i]=TopoGraph.connectivity_of_2_clusters(i,j,K_d,Dis,BoundaryMat_E,X_extend,real_manifolds)
        return ConnectMat
    
    
    #####################################################################
    #############################cut topo graph##########################
    @classmethod
    def cut_graph(self,W,ratio):
        W=W.copy()
        filter_mask=W/np.max(W,axis=1)<ratio
        not_mask=(np.sum(W>0,axis=1)<=2).reshape(-1,1)
        filter_mask=filter_mask*(~not_mask)
        W[filter_mask]=0
        W[filter_mask.T]=0
        return W