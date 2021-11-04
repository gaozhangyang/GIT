import numpy as np
from scipy.optimize import linprog

PCA_MNIST_GIT = [0.478355, 0.691515, 0.77153, 0.599865,0.551863]
PCA_MNIST_KM = [0.419655,0.498879,0.467717,0.501209,0.503281]
PCA_MNIST_SA = [0.305144,0.390322,0.36481,0.388679,0.399094]
PCA_MNIST_Q = [0.245804,0.646372,0.486525,0.541622,0.45017]


PCA_FMNIST_GIT = [0.592528,0.686799,0.643216,0.537525,0.551863]
PCA_FMNIST_KM = [0.492665,0.515958,0.387934,0.3879,0.388148]
PCA_FMNIST_SA = [0.435895,0.469618,0.419684,0.467485,0.473277]
PCA_FMNIST_Q = [0.422668,0.424779,0.402471,0.382822,0.41524]


AE_MNIST_GIT = [0.878381,0.861547,0.880494,0.842948,0.520227]
AE_MNIST_KM = [0.721353,0.775986,0.519486,0.485933,0.503281]
AE_MNIST_SA = [0.654951,0.621669,0.501961,0.424814,0.399094]
AE_MNIST_Q = [0.779808,0.739898,0.785197,0.680408,0.45017]


AE_FMNIST_GIT = [0.647393,0.578947,0.577126,0.650498,0.520227]
AE_FMNIST_KM = [0.39582,0.546557,0.527729,0.503655,0.388148]
AE_FMNIST_SA = [0.470541,0.493084,0.477636,0.378715,0.473277]
AE_FMNIST_Q = [0.345888,0.4461,0.384507,0.370655,0.41524]


def wasserstein_distance(p, q):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
    p = p/p.sum()
    q = q/q.sum()
    D = np.sqrt(np.square(p[:, None] - q[None, :]))
    # D = np.ones((p.shape[0],q.shape[0]))

    A_eq = []
    for i in range(len(p)):
        A = np.zeros_like(D)
        A[i, :] = 1
        A_eq.append(A.reshape(-1))
    for i in range(len(q)):
        A = np.zeros_like(D)
        A[:, i] = 1
        A_eq.append(A.reshape(-1))
    A_eq = np.array(A_eq)
    b_eq = np.concatenate([p, q])
    D = D.reshape(-1)
    result = linprog(D, A_eq=A_eq[:-1], b_eq=b_eq[:-1])
    return result.fun


class TopoGraph:
    def __init__(self):
        pass

    def graph_construction(self,V,X_extend,Boundary,Dis):
        E={}
        X_dis=[]
        for (i,j,ri,rj) in Boundary:
            E[(ri,rj)]=0
            E[(rj,ri)]=0
            X_dis.append( np.linalg.norm(X_extend[i,:-4]-X_extend[j,:-4]) )

        if len(X_dis)==0:
            RT=list(V.keys())
            return {(RT[0],RT[0]):0}

        for idx,(i,j,ri,rj) in enumerate(Boundary):
            s= np.exp(-X_dis[idx])/(len(V[ri])*len(V[rj]))
            E[(ri,rj)]+=s
            E[(rj,ri)]=E[(ri,rj)]
        return E

    def graph_pruning(self,E_raw,V,target,DataNumber):
        # TODO 当E_raw为空时，报错
        # print(E_raw)
        edges=[(i,j,v) for (i,j),v in E_raw.items()]
        edges=sorted(edges,key=lambda x:x[2],reverse=True)
        C=[{i:i for i in V.keys()}] # node_idx: cls_idx
        M=[{i:[i] for i in V.keys()}] # cls_idx: [node_idx]
        N=[{i:len(V[i]) for i in V.keys()}] # number of points in each cluster
        score=[9999]
        c=len(target)
        target = np.array(target)/np.array(target).sum()*DataNumber
        useless_edges = []
        E_final={}

        for i,j,v in edges:
            C_prev=C[-1]
            M_prev=M[-1]
            N_prev=N[-1]
            if C_prev[i]!=C_prev[j]:
                S_i=M_prev[i]
                S_j=M_prev[j]
                S=S_i+S_j
                
                C_now=C_prev.copy()
                for k in S_j:
                    C_now[k]=C_prev[i]
                
                M_now=M_prev.copy()
                for k in S:
                    M_now[k]=S
                    
                N_now=N_prev.copy()
                for k in S:
                    N_now[k]=N_prev[i]+N_prev[j]
                
                seq = np.array(sorted([N_now[c] for c in set(C_now.values())],reverse=True))
                if len(seq)<c:
                    break
                # sc = word_mover_distance(seq[:c],target)
                # sc = wasserstein_distance(seq[:c],target)
                sc = wasserstein_distance(seq,target)
                if sc<=score[-1]:
                    score.append(sc)
                    C.append(C_now)
                    M.append(M_now)
                    N.append(N_now)
                    E_final[(i,j)]=v
                else:
                    useless_edges.append((i,j,v))
        
        cls_set = list(set(C[-1].values()))
        classes = {idx:M[-1][root_idx] for idx,root_idx in enumerate(cls_set) } # ordered number-->index of local clusters in the same classes, e.g., 1:[2658, 2691, 2579, 2556, 2258, 2645]
        
        point_numbers = {idx:N[-1][root_idx] for idx,root_idx in enumerate(cls_set) }
        root2order = { root_idx:idx for idx,root_idx in enumerate(cls_set) }
        
        A = np.zeros((len(cls_set),len(cls_set)))
        for i,j,v in useless_edges:
            row = root2order[C[-1][i]]
            col = root2order[C[-1][j]]
            A[row,col]+=v
            A[col,row]+=v
        
        N_pred = np.array(list(point_numbers.values()))
        merge_idx = np.argsort(N_pred)
        target_set = merge_idx[-c:].tolist()
        source_set = merge_idx[:-c].tolist()
        A[:,target_set]=0
        target_idx, source_idx = A.nonzero()
        edges = [[i,j,A[i,j]] for i in target_idx for j in source_idx]
        edges = sorted(edges,key=lambda x: x[2],reverse=True)

        re_direct = {i:i for i in range(A.shape[0])}
        for target, source, value in edges:
            if len(source_set)==0:
                break

            if source in source_set:
                if re_direct[target]!=source:
                    classes[re_direct[target]]+=classes[source]
                    source_set.remove(source)
                    del classes[source]
                    
                    for key,val in re_direct.items():
                        if val == source:
                            re_direct[key]=re_direct[target]

        return E_final,classes
    
    def topograph_construction_pruning(self,V,X_extend,Boundary,Dis,target):
        E_raw = self.graph_construction(V,X_extend,Boundary,Dis)
        E_final,node2cls = self.graph_pruning(E_raw,V,target,X_extend.shape[0])
        return E_raw,E_final,node2cls
