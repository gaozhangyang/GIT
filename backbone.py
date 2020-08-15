import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import networkx as nx
from detect_local_mode import Manifold
import plot_tools
from pathlib import Path

class Backbone:
    def __init__(self):
        pass
    
    @classmethod
    def mean_center(self,X_extend,idx):
        left,right=idx[:,0],idx[:,1]
        new_pos=[]
        for dim in range(X_extend.shape[1]-2):
            new_pos.append(((X_extend[left,dim]+X_extend[right,dim])/2).reshape(-1,1))
        return np.hstack(new_pos)

    @classmethod
    def connectivity(self,i,j,K_d,Dis,BoundaryMat_E,X_extend,manifolds):
        BE=BoundaryMat_E[i,j]
        if BE.shape[0]==0:
            return 0
        P_mid,D,I=Dis.get_density(Backbone.mean_center(X_extend,BE),K_d+1,train=True)
        # P_left,D,I=Dis.get_density(X_extend[BE[:,0],:-2],K_d+1,train=True)
        # P_right,D,I=Dis.get_density(X_extend[BE[:,1],:-2],K_d+1,train=True)
        # P=np.min( np.hstack([P_left.reshape(-1,1),P_mid.reshape(-1,1),P_right.reshape(-1,1)]),axis=1 )
        # P = P_left*P_mid*P_right
        P=P_mid

        P1=manifolds[i].center[-2]#np.mean(manifolds[i].points[:,-2])
        P2=manifolds[j].center[-2]#np.mean(manifolds[j].points[:,-2])
        return np.sum(P**2)*min(P1/P2,P2/P1)**2
    
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
    def cut_graph(self,W,ratio):
        W=W.copy()
        filter_mask=W/np.max(W,axis=1)<ratio
        not_mask=(np.sum(W>0,axis=1)<=2).reshape(-1,1)
        filter_mask=filter_mask*(~not_mask)
        W[filter_mask]=0
        W[filter_mask.T]=0
        return W
    
    @classmethod
    def show_graph(self,W,manifolds,fileroot=None,fileidx=1000):
        plt.figure(figsize=(4, 4))
        tmp_G=nx.from_numpy_matrix(W)
        pos={i:(manifolds[i].center[0],manifolds[i].center[1]) for i in range(len(manifolds)) }
        nx.draw_networkx(tmp_G,pos)
        edge_labels = nx.get_edge_attributes(tmp_G, 'weight')
        edge_labels={key:'{:.2f}'.format(val) for key,val in edge_labels.items()}
        nx.draw_networkx_edge_labels(tmp_G, pos, edge_labels=edge_labels)
        plt.axis('equal')
        if fileroot:
            for i in range(5):
                plt.savefig(Path(fileroot.format(fileidx+i)))
    
    
    @classmethod
    def show_with_set(self,sets,manifolds,fileroot=None,fileidx=1000,title=None):
        colors={}
        colors.update({i:'C{}'.format(i) for i in range(len(sets))})
        plt.figure(figsize=(4, 4))
        for i_cls,oneset in enumerate(sets):
            for i_M in list(oneset):
                plt.scatter(manifolds[i_M].points[:,0],manifolds[i_M].points[:,1],c=colors[i_cls])
#                 plt.text(manifolds[i_M].center[0],manifolds[i_M].center[1],'{:.0f}'.format(i_M))
        plt.axis('equal')
        if title:
            plt.title(title)
        if fileroot:
            for i in range(15):
                plt.savefig(Path(fileroot.format(fileidx+i)))
        plt.show()
        
    
    @classmethod
    def fit(self,X,K_d,search_n,level=0.99,ratio=0,pnum=8,fps=2,mp4=False,figroot='./figs',mp4name='circles'):
        extend=np.zeros((X.shape[0],2))
        extend[:,1]=range(0,X.shape[0])
        X_extend=np.hstack([X,extend])
        
        ############KDE//noise detecting//local mode###########
        # show raw data
        plot_tools.autoPlot(X_extend[:,:-2],np.zeros(X_extend.shape[0]).astype(np.int))

        Dis,manifolds,connection,noise,P2M,draw_tasks=Manifold.get_manifolds(X_extend,K_d,level,search_n)
        # show local modes
        Manifold.show(manifolds)

        noise_manifold=manifolds[0] # noise manifold
        manifolds=manifolds[1:] # useful manifold

        ######################topological graph###############
        # construct topological graph
        connection=np.array(connection)
        BoundaryMat_E=Backbone.get_boundary(connection,manifolds)
        ConnectMat=np.zeros([len(manifolds),len(manifolds)])
        for i in range(len(manifolds)):
            for j in range(i+1,len(manifolds)):
                ConnectMat[i,j]=ConnectMat[j,i]=Backbone.connectivity(i,j,K_d,Dis,BoundaryMat_E,X_extend,manifolds)
        # show topological graph
        Backbone.show_graph(ConnectMat,manifolds)

        #########################prune graph#############
        W=Backbone.cut_graph(ConnectMat,ratio)
        # show Backbone
        Backbone.show_graph(W,manifolds)

        tmp_G=nx.from_numpy_matrix(W)
        Sets=list(nx.connected_components(tmp_G))
        # show clustering results
        Backbone.show_with_set(Sets,manifolds)

        ##########################pred###################
        Y_=np.zeros(X_extend.shape[0])
        NC=len(manifolds)
        for i in range(NC):
            points=manifolds[i].points
            M=points.shape[0]
            Y_[points[:,-1].astype(np.int)]=i
        Y_[noise]=-1
            
        if mp4:
            ploter=plot_tools.Visualization(figroot)
            ploter.run(pnum,draw_tasks)

            plt.figure(figsize=(4, 4))
            ax = plt.subplot(111)
            ax.scatter(X_extend[:,0], X_extend[:,1],c=X_extend[:,-2])
            plt.axis('equal')
            for idx in range(5):
                plt.savefig(figroot+'/{}.png'.format(idx-100))
            plt.close()
            
            NC=len(manifolds)
            color_list=['C{}'.format(i) for i in range(NC)]
            
            plt.figure(figsize=(4, 4))
            for i in range(NC):
                points=manifolds[i].points
                centers=manifolds[i].center
                plt.scatter(points[:,0],points[:,1],c=color_list[i])
                plt.text(centers[0],centers[1],str(i))
            plt.axis('equal')
            for idx in range(5):
                plt.savefig(figroot+'/{}.png'.format(10000000+idx))
            plt.close()

            Backbone.show_graph(ConnectMat,manifolds,fileroot=figroot+'/{}.png',fileidx=20000000)
            Backbone.show_graph(W,manifolds,fileroot=figroot+'/{}.png',fileidx=30000000)
            Backbone.show_with_set(Sets,manifolds,fileroot=figroot+'/{}.png',fileidx=40000000,title='k:{} search_n:{} ratio:{}'.format(k,search_n,ratio))
            ploter.SaveGIF(mp4name,fps=fps)

        return X_extend[:,:-2],Y_,X_extend[:,-2],W