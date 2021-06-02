import numpy as np
from scipy.optimize import linprog

def wasserstein_distance(p, q, D):
    """通过线性规划求Wasserstein距离
    p.shape=[m], q.shape=[n], D.shape=[m, n]
    p.sum()=1, q.sum()=1, p∈[0,1], q∈[0,1]
    """
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

def word_mover_distance(x, y):
    """WMD（Word Mover's Distance）的参考实现
    x.shape=[m,d], y.shape=[n,d]
    """
    p = np.ones(x.shape[0]) #/ x.shape[0]
    q = np.ones(y.shape[0]) #/ y.shape[0]
    D = np.sqrt(np.square(x[:, None] - y[None, :]))
    return wasserstein_distance(p, q, D)


def optimum_params(E_raw,V,target,DataNumber):
    edges=[(i,j,v) for (i,j),v in E_raw.items()]
    edges=sorted(edges,key=lambda x:x[2],reverse=True)
    F=[{i:i for i in V.keys()}] # node_idx: cls_idx
    G=[{i:[i] for i in V.keys()}] # cls_idx: [node_idx]
    N=[{i:len(V[i]) for i in V.keys()}] # number of points in each cluster
    W=[1]

    for i,j,w in edges:
        f_prev=F[-1]
        g_prev=G[-1]
        n_prev=N[-1]
        if f_prev[i]!=f_prev[j]:
            S_i=g_prev[i]
            S_j=g_prev[j]
            S=S_i+S_j
            
            f_now=f_prev.copy()
            for k in S_j:
                f_now[k]=f_prev[i]
            
            g_now=g_prev.copy()
            for k in S:
                g_now[k]=S
                
            n_now=n_prev.copy()
            for k in S:
                n_now[k]=n_prev[i]+n_prev[j]
            
            F.append(f_now)
            G.append(g_now)
            N.append(n_now)
            W.append(w)

    c=len(target)
    seqs=[]
    weights=[]
    for idx in range(len(N)):
        seq = np.array(sorted([N[idx][c] for c in set(F[idx].values())],reverse=True))
        if len(seq)<c:
            break
        seqs.append(seq[:c])
        weights.append(W[idx])
    
    score=[]
    target = np.array(target)/np.array(target).sum()*DataNumber
    for idx in range(len(seqs)):
        score.append(word_mover_distance(seqs[idx],target))

    k=np.argmin(score)
    best_threshold = weights[k]
    return best_threshold,weights[1:],score[1:],F[k]