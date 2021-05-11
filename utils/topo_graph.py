import numpy as np

# class TopoGraph:
#     def __init__(self):
#         pass

#     def topograph_construction_pruning(self,V,X_extend,Boundary,Dis,alpha,epsilon):
#         E,E_hat,gamma,count={},{},{},{}
#         X_mid=[]
#         for (i,j,ri,rj) in Boundary:
#             E[(ri,rj)]=0
#             E[(rj,ri)]=0
#             E_hat[ri]=0
#             E_hat[rj]=0
#             gamma[(ri,rj)]=None
#             count[(ri,rj)]=0
#             X_mid.append( (X_extend[i,-3],X_extend[j,-3]) )

#         if len(X_mid)==0:
#             return E.copy(),E

#         X_mid=np.array(X_mid).astype(np.int)
#         # P_left=Dis.get_density(X_mid[:,0])
#         # P_right=Dis.get_density(X_mid[:,1])
#         # P_mid = (P_left+P_right)/2

#         # P_mid = np.exp(-(np.linalg.norm(X_extend[X_mid[:,0],:-4]-X_extend[X_mid[:,1],:-4],axis=1)/2/np.sqrt(X_extend.shape[1]-4))**2)
#         P_mid = Dis.get_midP(X_mid[:,0],X_mid[:,1])


#         for idx,(i,j,ri,rj) in enumerate(Boundary):
#             if gamma[(ri,rj)] is None:
#                 P1,P2=X_extend[ri,-4],X_extend[rj,-4]
#                 gamma[(ri,rj)]=gamma[(rj,ri)]=min(P1/P2,P2/P1)**2#/max(P2,P1)**2
            
#             s= (gamma[(ri,rj)]*(P_mid[idx])**2) / (len(V[ri])*len(V[rj]))
#             count[(ri,rj)]+=1
#             count[(rj,ri)]=count[(ri,rj)]
#             E[(ri,rj)]+=s
#             E[(rj,ri)]=E[(ri,rj)]

#             if E[(ri,rj)]>E_hat[ri]:
#                 E_hat[ri]=E[(ri,rj)]

#             if E[(ri,rj)]>E_hat[rj]:
#                 E_hat[rj]=E[(ri,rj)]

#         E_raw=E.copy()
#         for key,value in E.items():
#             i,j=key[0],key[1]
#             if value<=alpha*E_hat[i] or value<=alpha*E_hat[j]:
#                 E[(i,j)]=0
#                 E[(j,i)]=0
        
#         weights = sorted([one for one in E.values() if one>0])
#         threshold = weights[int(len(weights)*alpha)]

#         for key,value in E.items():
#             i,j=key[0],key[1]
#             if value<=threshold:
#                 E[(i,j)]=0
#                 E[(j,i)]=0
#         return E_raw,E


import scipy
from sklearn.cluster import KMeans

class TopoGraph:
    def __init__(self,n_clusters,V,alpha=0.2):
        super(TopoGraph,self).__init__()
        self.alpha=alpha
        self.clusterer = KMeans(n_clusters=n_clusters)
        V = sorted(list(V.keys()))
        self.idx_point2graph = {V[g]:g for g in range(len(V))}
        self.idx_graph2point = {g:V[g] for g in range(len(V))}

        roots = np.array(V).astype(np.int)
        self.idx_point2graph_arr = np.zeros(np.max(roots)+1)-1
        self.idx_point2graph_arr[roots]=np.arange(roots.shape[0])
    
    # def idx_point2graph(self,V):
    #     v = np.array(sorted(list(V.keys()))).reshape(-1,1)
    #     results = (v == self.roots).astype(int).sum(dim=0)
    #     index = np.nonzero(results)
    #     return index
    
    # def idx_graph2point(self,Index):
    #     pass

    def topograph_construction_pruning(self,V,X_extend,Boundary,Dis):
        E,E_hat,gamma,count={},{},{},{}
        X_mid=[]
        for (i,j,ri,rj) in Boundary:
            ri, rj = self.idx_point2graph[ri],self.idx_point2graph[rj]
            E[(ri,rj)]=0
            E[(rj,ri)]=0
            E_hat[ri]=0
            E_hat[rj]=0
            gamma[(ri,rj)]=None
            count[(ri,rj)]=0
            X_mid.append( (X_extend[i,-3],X_extend[j,-3]) )

        if len(X_mid)==0:
            return E.copy(),E

        X_mid=np.array(X_mid).astype(np.int)
        P_mid = Dis.get_midP(X_mid[:,0],X_mid[:,1])

        Used_Bound_P=[]
        for idx,(i,j,ri,rj) in enumerate(Boundary):
            if i in Used_Bound_P:
                continue
            if j in Used_Bound_P:
                continue
            Used_Bound_P.append(i)
            Used_Bound_P.append(j)
            ri0, rj0 = ri,rj
            ri, rj = self.idx_point2graph[ri],self.idx_point2graph[rj]
            if gamma[(ri,rj)] is None:
                P1,P2=X_extend[ri0,-4],X_extend[rj0,-4]
                gamma[(ri,rj)]=gamma[(rj,ri)]=min(P1/P2,P2/P1)**2#/max(P2,P1)**2
            
            # s= (gamma[(ri,rj)]*(P_mid[idx])**2) #/ (len(V[ri0])*len(V[rj0]))
            P1,P2=X_extend[ri0,-4],X_extend[rj0,-4]
            s = ((P1+P2)/2)**2*gamma[(ri,rj)]#/ (len(V[ri0])*len(V[rj0]))
            count[(ri,rj)]+=1
            count[(rj,ri)]=count[(ri,rj)]

            E[(ri,rj)]+=s
            E[(rj,ri)]=E[(ri,rj)]

            if E[(ri,rj)]>E_hat[ri]:
                E_hat[ri]=E[(ri,rj)]

            if E[(ri,rj)]>E_hat[rj]:
                E_hat[rj]=E[(ri,rj)]

        alpha = self.alpha
        E_raw=E.copy()
        for key,value in E.items():
            i,j=key[0],key[1]
            if value<=alpha*E_hat[i] or value<=alpha*E_hat[j]:
                E[(i,j)]=0
                E[(j,i)]=0
        
        weights = sorted([one for one in E.values() if one>0])
        threshold = weights[int(len(weights)*alpha)]

        for key,value in E.items():
            i,j=key[0],key[1]
            if value<=threshold:
                E[(i,j)]=0
                E[(j,i)]=0


        G = np.zeros((len(V),len(V)))
        for (i,j),v in E.items():
            G[i,j]=v
            G[j,i]=v
        
        # d = np.sum(G, axis=1)
        # d[d == 0] = 1
        # d = np.power(d, -0.5)
        # D = scipy.sparse.diags(np.squeeze(np.asarray(d)))
        # G = D @ G @ D
        
        Lambda, Vec = scipy.sparse.linalg.eigsh(G, k=30, which="LM")
        Lambda, Vec = np.absolute(Lambda), np.absolute(Vec)

        base_labels = self.clusterer.fit_predict(Vec * np.power(Lambda, 0.5))

        idx = self.idx_point2graph_arr[X_extend[:,-2].astype(np.int)].astype(np.int)
        Y = base_labels[idx]
        return Y,E