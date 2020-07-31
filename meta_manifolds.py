import copy
import scipy
from scipy.spatial import distance_matrix
from sklearn.preprocessing import normalize
from binarytree import tree,Node
import random
import numpy as np
from KDE import Point_Distance
import matplotlib.pyplot as plt
        
class Union_Search():
    def __init__(self,n):
        self.pre=np.arange(n)# 记录n个点的父亲节点
        
        
    def find(self,x,pre=None):
        if pre is None:
            pre=self.pre
        
        rt=x
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
            
        B_mask=np.ones_like(root)
        for x in range(root.shape[0]):
            rt=US.find(x,root)
            now=x
            while now!=rt:
                parent=root[now]
                root[now]=rt
                now=parent
                B_mask[parent]=0
                
        return root,B_mask 


class Manifold:
    def __init__(self):
        self.points=np.zeros((10,4))#倒数第二维:P 最后一维:ID
        self.center=np.mean(self.points,axis=0)
        self.name=''
    
    def update(self,points,name):
        self.points=points
        self.center=np.mean(self.points,axis=0)
        self.name=name
    
    @classmethod
    def detect_descending_manifolds(self,X,D,I,k,isShow=False,Union=False):
        f=X[:,-2]
        idx=np.argsort(f).tolist()
        idx.reverse() #从大到小排序f
        US=Union_Search(D.shape[0])# D:(N,k) 初始化Union_Search结构，共N个点
        index=0
        connection=[]
        draw_tasks=[]
        draw_step=int(X.shape[0]//100)
        while len(idx)>0:
            index+=1
            i=idx[0] #取密度最高的点i
            
            idx_iN=I[i,1:k+1] # 距离约束，取k近邻的索引值
            tmp_idx=idx_iN[f[idx_iN]>f[i]] # 对k近邻施加密度约束,使得选择的近邻点的密度都大于当前点

            if tmp_idx.shape[0]>0:
                j=tmp_idx[0] # 父亲节点被找到
            else:
                j=None # 父亲节点没有被找到

            if j is not None: # 父亲节点被找到
                US.join(j,i) # i的父亲是j，建立连接
            
            if index%draw_step==0:
                draw_tasks.append( (X[:,:2],US.flatten(US),index) )

            if tmp_idx.shape[0]>1:
                rt0=US.find(tmp_idx[0])
                for query in tmp_idx[1:]:
                    rt1=US.find(query)
                    if rt0!=rt1:
                        # print(f[i],min(f[i]/f[rt0],f[i]/f[rt1]))
                        certainty=np.sum(f[I[i]]<f[i])/I.shape[1]
                        # print(certainty)
                        connection.append( (rt0,rt1,certainty*(f[i])**2) )

            idx.remove(i) #删除密度最高的点i
        
        
        root,B_mask=US.flatten(US,US.pre)
        manifolds=[]
        P2M={}
        N=0
        for i in list(set(root)):
            M=Manifold()
            index=np.nonzero(root==i)[0]
            M.update(X[index],str(N))
            P2M.update({int(point):N for point in X[index,-1]})
            manifolds.append(M)
            N+=1
        
        M_connection=[]
        for i,j,p in connection:
            M_connection.append((P2M[i],P2M[j],p))
            
        return manifolds,M_connection,P2M,draw_tasks
    
    @classmethod
    def show(self,manifolds,mode='points'):
        NC=len(manifolds)
        color_list=['C{}'.format(i) for i in range(NC)]
        
        plt.figure(figsize=(4, 4))
        if mode=='points':
            for i in range(NC):
                points=manifolds[i].points
                centers=manifolds[i].center
                plt.scatter(points[:,0],points[:,1],c=color_list[i])
                plt.text(centers[0],centers[1],str(i))
                
        if mode=='center':
            for i in range(NC):
                points=manifolds[i].center
                plt.scatter(points[0],points[1],c=color_list[i])
                plt.text(points[0],points[1],str(i))
        
        _=plt.axis('equal')
        plt.show()
    
    @classmethod
    def augumentation(self,manifolds,drop=0.2,noise=0.1):
        points=[]
        pIDs=[]
        Ps=[]
        new_manifolds=[]
        for M in manifolds:
            point=M.points.copy()
            pID=M.pID.copy()
            M_P=M.P.copy()
            
            th=np.argsort(M.P)[int(drop*point.shape[0])]
            mask=M.P>M.P[th]
            new_point=point[mask]+np.random.random(point[mask].shape)*noise
            points.append(new_point)
            pIDs.append(pID[mask])
            Ps.append(M_P[mask])
            
            newM=Manifold()
            newM.points=new_point
            newM.pID=pID[mask]
            newM.P=M_P[mask]
            new_manifolds.append(newM)
            
        points=np.vstack(points)
        pIDs=np.hstack(pIDs)
        return new_manifolds,points,pIDs
    
    @classmethod
    def get_scale(self,X):
        x=X[:,:2]
        sigma=np.sqrt(np.var(x))
        scale=(4*sigma**5/(3*x.shape[0]))**0.2
        return scale

    
    @classmethod
    def get_manifolds(self,X,k,search_n=30):
        isShow=False
        Union=False

        Dis=Point_Distance(X[:,:2],param=k+1)
        D, I=Dis.get_DI(X[:,:2],param=k+1)#D,I:(N,k)  D:distance of neighbors I: index of neighbors
        P=Dis.get_density(D) #计算点的密度
        P=(P-np.min(P))/(np.max(P)-np.min(P))
        X[:,-2]=P
        manifolds,M_connection,P2M,draw_tasks=Manifold.detect_descending_manifolds(X,D,I,k=search_n,isShow=isShow,Union=Union)
        return manifolds,M_connection,P2M,draw_tasks