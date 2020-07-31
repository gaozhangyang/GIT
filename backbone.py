import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import numpy as np
import networkx as nx
from meta_manifolds import Manifold
import plot_tools
from pathlib import Path

class Backbone:
    def __init__(self):
        pass
    
    @classmethod
    def distance_Ms(self, manifolds, M_connection, level, use_P=True):
        epsilon=1e-10
        M_num=len(manifolds)
        W=np.zeros((M_num,M_num))
        M_connection=sorted(M_connection,key=lambda x:x[2],reverse=True)
        for i in range(len(M_connection)):
            if M_connection[i][2]>level:
                M1,M2,P=M_connection[i]
                W[M1,M2]+=P

                if use_P:
                    PM1=np.mean(manifolds[M1].points[:,-2])
                    PM2=np.mean(manifolds[M2].points[:,-2])
                    W[M1,M2]*=min(PM1/PM2,PM2/PM1)
                    
                W[M2,M1]=W[M1,M2]
        
        return W
    
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
            for i in range(5):
                plt.savefig(Path(fileroot.format(fileidx+i)))
        plt.show()
        
    
    @classmethod
    def fit(self,X,k,search_n,level=0,ratio=None,pnum=8,fps=2,mp4=False,figroot='./figs',mp4name='circles'):
        plot_tools.autoPlot(X[:,:2],np.zeros(X.shape[0]).astype(np.int))

        manifolds,M_connection,P2M,draw_tasks=Manifold.get_manifolds(X,k,search_n)
        Manifold.show(manifolds)

        W=Backbone.distance_Ms(manifolds,M_connection,level=level,use_P=True)
        W2=Backbone.cut_graph(W,ratio)
        Backbone.show_graph(W2,manifolds)

        tmp_G=nx.from_numpy_matrix(W2)
        Sets=list(nx.connected_components(tmp_G))
        Backbone.show_with_set(Sets,manifolds)

        if mp4:
            ploter=plot_tools.Visualization(figroot)
            ploter.run(pnum,draw_tasks)
            Backbone.show_graph(W2,manifolds,fileroot=figroot+'/{}.png',fileidx=10000000)
            Backbone.show_with_set(Sets,manifolds,fileroot=figroot+'/{}.png',fileidx=20000000,title='k:{} search_n:{} ratio:{}'.format(k,search_n,ratio))
            ploter.SaveGIF(mp4name,fps=fps)

        return W,W2,draw_tasks