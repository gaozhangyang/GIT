import numpy as np
from kde import KDE_DIS
import time
from tqdm import tqdm
        
class LCluster:
    def __init__(self):
        pass

    @classmethod
    def detect_descending_manifolds(self,X,K_d,K_s):
        extend=np.arange(0,X.shape[0]).reshape(-1,1).repeat(4,axis=1) # extend 4 dim, representing $density $ID $local clusters $ label
        X_extend=np.hstack([X,extend])

        t0=time.time()
        Dis=KDE_DIS(X_extend[:,:-4],K=K_d)
        D,I = Dis.get_DI(X_extend[:,-3])
        P = Dis.get_density(X_extend[:,-3])
        t2=time.time()
        X_extend[:,-4]=P

        idx=np.argsort(P).tolist()
        idx.reverse()
        Boundary=[]

        Back_mask = np.ones(X.shape[0])
        draw_step=int(X.shape[0]//100)
        index=0
        draw_tasks=[]

        
        while len(idx)>0:
            i=idx[0] # Get the point with the highest current density, i
            index+=1
            Back_mask[i]=0

            idx_iN=I[i,1:K_s+1] # the index value nearest k neighbors
            mask = P[idx_iN]>P[i]
            J=idx_iN[mask] # the density of the selected neighboring points is greater than the current point

            if J.shape[0]>0:
                grad=(P[J]-P[i])/(D[i,1:K_s+1][mask])
                j=J[np.argmax(grad)]
            else:
                j=None # The father node was not found

            if j is not None: # The father node was found
                X_extend[i,-2]=X_extend[j,-2]
                for s in J[1:]:
                    if X_extend[s,-2]!=X_extend[i,-2]:
                        Boundary.append((i,s,int(X_extend[i,-2]),int(X_extend[s,-2])))
            
            if index%draw_step==0:
                draw_tasks.append( (X[:,:2],X[:,-2],Back_mask.copy(),index) ) # points, local clusters, Back_mask, time index
            
            idx.remove(i) # delete point i
        
        R=X_extend[:,-2]
        V={int(j):[] for j in set(R)}
        for i in range(len(R)):
            V[R[i]].append(i)
        
        return V,Boundary,X_extend,Dis,draw_tasks,t2