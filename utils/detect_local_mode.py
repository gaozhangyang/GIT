import numpy as np
from kde import KDE_DIS
        
class Union_Search():
    def __init__(self,n):
        self.pre=np.arange(n)# 记录n个点的父亲节点
        
        
    def find(self,x,pre=None):
        if pre is None:
            pre=self.pre
        
        rt=x
        if pre[rt]==-1:
            return -1

        while pre[rt]!=rt:
            rt=pre[rt]
            
        return rt
    
    
    def join(self,x,y):
        self.pre[y]=x
    
    
    def flatten(self,US,root=None):
        if root is None:
            root=self.pre.copy()
        else:
            root=root.copy()
            
        for x in range(root.shape[0]):
            rt=US.find(x,root)
            now=x
            while now!=rt:#把沿线上的point指向rt
                parent=root[now]
                root[now]=rt
                now=parent
                
        return root 


class Manifold:
    def __init__(self):
        self.points=np.zeros((10,4))#倒数第二维:P 最后一维:ID
        self.center=np.mean(self.points,axis=0)
        self.name=''
        self.rt=None
    
    def update(self,points,name,rt):
        self.points=points
        self.center=np.mean(self.points,axis=0)
        self.name=name
        self.rt=rt
    
    @classmethod
    def detect_descending_manifolds(self,X,D,I,K_d,level,isShow=False,Union=False):
        f=X[:,-2]
        idx=np.argsort(f).tolist()
        idx.reverse() #从大到小排序f
        US=Union_Search(D.shape[0])# D:(N,k) 初始化Union_Search结构，共N个点
        index=0
        connection=[]
        draw_tasks=[]
        noise=[]
        draw_step=int(X.shape[0]//100)
        while len(idx)>0:
            index+=1
            i=idx[0] #取密度最高的点i
            
            idx_iN=I[i,1:K_d+1] # 距离约束，取k近邻的索引值
            mask = f[idx_iN]>f[i]
            tmp_idx=idx_iN[mask] # 对k近邻施加密度约束,使得选择的近邻点的密度都大于当前点

            if tmp_idx.shape[0]>0:
                grad=(f[tmp_idx]-f[i])/(D[i,1:K_d+1][mask])
                j=tmp_idx[np.argmax(grad)]
            else:
                j=None # 父亲节点没有被找到

            if j is not None: # 父亲节点被找到
                rt0=US.find(j)
                if f[rt0]-f[i]>level*f[rt0]:
                    noise.append(i)
                else:
                    US.join(j,i) # i的父亲是j，建立连接
            
            if index%draw_step==0:
                draw_tasks.append( (X[:,:2],US.flatten(US),index) )

            
            if tmp_idx.shape[0]>1 and US.pre[i]!=-1:
                if f[rt0]-f[i]<=level*f[rt0]:
                    # rt0=US.find(tmp_idx[0]) # i=tmp_idx[0]
                    for query in tmp_idx[1:]:
                        rt1=US.find(query)
                        if rt0!=rt1:
                            connection.append( (i,query,rt0,rt1) )
            idx.remove(i) #删除密度最高的点i
        
        root=US.flatten(US,US.pre)
        for noise_i in noise:
            root[root==noise_i]=-1

        manifolds=[]
        P2M={}
        N=-1
        for i in sorted(list(set(root))):
            M=Manifold()
            index=np.nonzero(root==i)[0]
            M.update(X[index],str(N),i)
            P2M.update({int(point):N for point in X[index,-1]})
            manifolds.append(M)
            N+=1
            
        return manifolds,connection,noise,P2M,draw_tasks
    
    @classmethod
    def get_manifolds(self,X_extend,K_d,level,search_n=30):
        isShow=False
        Union=False

        Dis=KDE_DIS(X_extend[:,:-2],param=K_d+1)
        P,D,I =Dis.get_density(X_extend[:,:-2],K_d+1,train=True) #计算点的密度
        P=(P-np.min(P))/(np.max(P)-np.min(P)+1e-10)
        X_extend[:,-2]=P
        manifolds,M_connection,noise,P2M,draw_tasks=Manifold.detect_descending_manifolds(X_extend,D,I,K_d=search_n,level=level,isShow=isShow,Union=Union)
        return Dis,manifolds,M_connection,noise,P2M,draw_tasks