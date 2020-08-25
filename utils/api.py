from networkx.readwrite.json_graph import tree
import numpy as np
import networkx as nx
from numpy import random
from detect_local_mode import Manifold
from topo_graph import TopoGraph
import plot_tools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import pickle

class DGSFC:
    def __init__(self) -> None:
        pass

    @classmethod
    def fit(self,X,
                 K_d,
                 K_n,
                 epsilon=1,
                 lamda=0,
                 plot=False,
                 pnum=8,
                 fps=2,
                 mp4=False,
                 figroot='./figs',
                 mp4name='circles',
                 draw_seed=2020
                 ):
        '''
        K_d: k neighbors to estimiate the local density
        K_n: K neighbors in density growing process, for estimate the gradient flows
        epsilon: the threshold for noise dropping
        lambda: the threshold for topo-graph pruning

        '''
                 
        extend=np.zeros((X.shape[0],2))
        extend[:,1]=range(0,X.shape[0])
        X_extend=np.hstack([X,extend])
        
        ############KDE//noise detecting//local mode###########
        Dis,manifolds,connection,noise,P2M,draw_tasks=Manifold.get_manifolds(X_extend,K_d,epsilon,K_n)
        if len(noise)==0:
            noise_manifold=None
            real_manifolds=manifolds
        else:
            noise_manifold=manifolds[0] # noise manifold
            real_manifolds=manifolds[1:] # useful manifold


        ######################topological graph###############
        # construct topological graph
        connection=np.array(connection)
        BoundaryMat_E=TopoGraph.get_boundary(connection,real_manifolds)
        ConnectMat=TopoGraph.connectivity_all(real_manifolds,K_d,Dis,BoundaryMat_E,X_extend)
        

        #########################prune graph#############
        W=TopoGraph.cut_graph(ConnectMat,lamda)
        tmp_G=nx.from_numpy_matrix(W)
        Sets=list(nx.connected_components(tmp_G))

        ##########################pred###################
        M2C={}
        for c in range(len(Sets)):
            for m in list(Sets[c]):
                M2C.update({m:c})
        
        classify={i:[] for i in range(len(Sets))}
        for m in range(len(real_manifolds)):
            classify[M2C[m]]+=real_manifolds[m].pID.tolist()
        
        if noise_manifold:
            classify.update({-1:noise_manifold.pID.tolist()})

        Y=np.zeros(X_extend.shape[0])
        for c in classify.keys():
            Y[classify[c]]=c
        Y=Y.astype(np.int)
        


        ########################plot######################
        if plot:
            Dim=X_extend.shape[1]
            if Dim<=4:
                # show raw data
                plot_tools.autoPlot(X_extend[:,:-2],np.zeros(X_extend.shape[0]).astype(np.int))
                # show local modes
                plot_tools.PaperGraph.show_manifolds(manifolds,X_extend)
                # show topo-graph
                plot_tools.PaperGraph.show_topo_graph(ConnectMat,real_manifolds,X_extend)
                # show pruned topo-graph
                plot_tools.PaperGraph.show_topo_graph(W,real_manifolds,X_extend)
                # show clustering results
                plot_tools.autoPlot(X_extend[:,:-2],Y)
            else:
                # show pruned topo-graph
                plot_tools.PaperGraph.show_topo_graph(W,real_manifolds)
                

            
        if mp4:
            ploter=plot_tools.Visualization(figroot,seed=draw_seed)
            ploter.run(pnum,draw_tasks)
            pickle.dump(draw_tasks,open(figroot+'/data.pkl','wb'))
            

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
                points= X_extend[ manifolds[i].pID,:-2 ]
                centers=np.mean(points,axis=0)
                plt.scatter(points[:,0],points[:,1],c=color_list[i])
                plt.text(centers[0],centers[1],str(i))
            plt.axis('equal')
            for idx in range(5):
                plt.savefig(figroot+'/{}.png'.format(10000000+idx))
            plt.close()

            plot_tools.PaperGraph.show_topo_graph(ConnectMat,real_manifolds,X_extend,fileroot=figroot+'/{}.png',fileidx=20000000)
            plot_tools.PaperGraph.show_topo_graph(W,real_manifolds,X_extend,fileroot=figroot+'/{}.png',fileidx=30000000)
            plot_tools.PaperGraph.show_clusters(X_extend[:,:-2],Y,fileroot=figroot+'/{}.png',fileidx=40000000,title='k:{} K_n:{} level:{} ratio:{}'.format(K_d,K_n,level,ratio))
            ploter.SaveGIF(mp4name,fps=fps)

        return Y