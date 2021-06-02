import numpy as np
from numpy.core.defchararray import mod
from scipy.optimize import linprog
from mip import Model, MINIMIZE, CBC, INTEGER, OptimizationStatus,BINARY,xsum,CONTINUOUS
import itertools

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
            return E.copy(),E
        
        # X_dis = np.array(X_dis)
        # X_mid=np.array(X_mid)#.astype(np.int)
        # P_mid = Dis.get_midP(X_mid[:,0],X_mid[:,1])

        for idx,(i,j,ri,rj) in enumerate(Boundary):
            s= np.exp(-X_dis[idx])/(len(V[ri])*len(V[rj]))
            E[(ri,rj)]+=s
            E[(rj,ri)]=E[(ri,rj)]
        
        return E

    def graph_pruning(self,E_raw,V,target,DataNumber):
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
                sc = word_mover_distance(seq[:c],target)
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

        for target, source, value in edges:
            if len(source_set)==0:
                break

            if source in source_set:
                classes[target]+=classes[source]
                source_set.remove(source)
                del classes[source]
            
            

        # for i in target_set:
        #     Connect=True
        #     while Connect:
        #         child = np.argmax(A[i])
        #         if A[i,child]>0:
        #             classes[i]+=classes[child]
        #             A[i,:] = A[i,:]+A[child,:]
        #             A[:,child]=-1
        #             del classes[child]
        #             source_set.remove(child)
        #         else:
        #             Connect=False

        # Rt=[0] # 寻找local cluster的根，相连的local cluster共享一个根
        # for i in range(1,len(cls_set)):
        #     k = np.nonzero(A[i])[0][0]
        #     if k<i:
        #         Rt.append(Rt[k])
        #     else:
        #         Rt.append(i)
        
        # connected_sets = [np.where(np.array(Rt)==r)[0].tolist() for r in set(Rt)]

        # model = Model(sense=MINIMIZE, solver_name=CBC)
        # c,c_hat = len(N_true),len(N_pred)
        # x = model.add_var_tensor((c,c_hat),name='x', var_type=INTEGER, lb=0, ub=1)
        # z = model.add_var_tensor((c,1),name='z', var_type=CONTINUOUS)
        # error = N_true-x@N_pred

        # model.objective = sum( z[i,0] for i in range(c))

        # for j in range(c_hat):
        #     model.add_constr( sum(x[i,j] for i in range(c)) == 1 )

        # for i in range(c):
        #     model.add_constr( -z[i,0]<=error[i,0] )
        #     model.add_constr(error[i,0]<=z[i,0])
        
        # for pro in itertools.product(*connected_sets):
        #     for i in range(c):
        #         model.add_lazy_constr( sum(x[i,k] for k in pro)<=1 )

        # status = model.optimize(max_seconds=2)
        # result = np.array([x[i,j].x for i in range(c) for j in range(c_hat)]).reshape(c,c_hat)

        # final_cluster = {i:[] for i in range(c)} # order number --> index of inner local cluster
        # for i in range(c):
        #     idx= np.nonzero(result[i,:])[0]
        #     for j in idx:
        #         final_cluster[i]+=classes[j]

        return E_final,classes
    
    def topograph_construction_pruning(self,V,X_extend,Boundary,Dis,target):
        E_raw = self.graph_construction(V,X_extend,Boundary,Dis)
        E_final,node2cls = self.graph_pruning(E_raw,V,target,X_extend.shape[0])
        return E_raw,E_final,node2cls



    # def topograph_construction_pruning(self,V,X_extend,Boundary,Dis,alpha,target_ratio):
    #     E={}
    #     X_mid=[]
    #     for (i,j,ri,rj) in Boundary:
    #         E[(ri,rj)]=0
    #         E[(rj,ri)]=0
    #         X_mid.append( (X_extend[i,-3],X_extend[j,-3]) )

    #     if len(X_mid)==0:
    #         return E.copy(),E

    #     X_mid=np.array(X_mid).astype(np.int)
    #     P_mid = Dis.get_midP(X_mid[:,0],X_mid[:,1])

    #     for idx,(i,j,ri,rj) in enumerate(Boundary):
    #         # if way=='P_mid':
    #         #     s = (P_mid[idx])**2
    #         # if way=='gamma*P_mid':
    #         #     s = (gamma[(ri,rj)]*(P_mid[idx])**2)
    #         # if way=='gamma*P_mid/V':
    #         #     s= (gamma[(ri,rj)]*(P_mid[idx])**2)/(len(V[ri])*len(V[rj]))
    #         # if way=='P_mid/V':
    #         #     s= (P_mid[idx])**2/(len(V[ri])*len(V[rj]))

    #         s= (P_mid[idx])**2/(len(V[ri])*len(V[rj]))
    #         E[(ri,rj)]+=s
    #         E[(rj,ri)]=E[(ri,rj)]

    #     E_raw=E.copy()
    #     if alpha is None:
    #         best_alpha,inter_alpha,inter_score,node2cls = optimum_params(E_raw,V,target_ratio,X_extend.shape[0])
    #     else:
    #         best_alpha = alpha

        
    #     for key,value in E.items():
    #         i,j=key[0],key[1]
    #         if value<best_alpha:
    #             E[(i,j)]=0
    #             E[(j,i)]=0
    #     return E_raw,E,best_alpha,inter_alpha,inter_score,node2cls