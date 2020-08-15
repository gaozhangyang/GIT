from networkx.readwrite.json_graph import tree
import numpy as np
import networkx as nx
from detect_local_mode import Manifold
from topo_graph import TopoGraph
import plot_tools
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

class DGSFC:
    def __init__(self) -> None:
        pass

    @classmethod
    def fit(self,X,
                 K_d,
                 search_n,
                 level=1,
                 ratio=0,
                 plot=False,
                 pnum=8,
                 fps=2,
                 mp4=False,
                 figroot='./figs',
                 mp4name='circles'
                 ):
                 
        extend=np.zeros((X.shape[0],2))
        extend[:,1]=range(0,X.shape[0])
        X_extend=np.hstack([X,extend])
        
        ############KDE//noise detecting//local mode###########
        Dis,manifolds,connection,noise,P2M,draw_tasks=Manifold.get_manifolds(X_extend,K_d,level,search_n)
        noise_manifold=manifolds[0] # noise manifold
        real_manifolds=manifolds[1:] # useful manifold


        ######################topological graph###############
        # construct topological graph
        connection=np.array(connection)
        BoundaryMat_E=TopoGraph.get_boundary(connection,real_manifolds)
        ConnectMat=TopoGraph.connectivity_all(real_manifolds,K_d,Dis,BoundaryMat_E,X_extend)
        

        #########################prune graph#############
        W=TopoGraph.cut_graph(ConnectMat,ratio)
        tmp_G=nx.from_numpy_matrix(W)
        Sets=list(nx.connected_components(tmp_G))

        ##########################pred###################
        X2=[]
        Y2=[]
        for c in range(len(Sets)):
            for idx in list(Sets[c]):
                X2.append(real_manifolds[idx].points[:,:-2])
                Y2.append(np.ones([real_manifolds[idx].points.shape[0],1])*c)
        X2.append(noise_manifold.points[:,:-2])
        Y2.append(-np.ones([noise_manifold.points.shape[0],1]))
        X2=np.vstack(X2)
        Y2=np.vstack(Y2).astype(np.int)


        ########################plot######################
        if plot:
            # show raw data
            plot_tools.autoPlot(X_extend[:,:-2],np.zeros(X_extend.shape[0]).astype(np.int))

            # show local modes
            plot_tools.PaperGraph.show_manifolds(manifolds)

            # show topological graph
            plot_tools.PaperGraph.show_topo_graph(ConnectMat,real_manifolds)

            # show TopoGraph
            plot_tools.PaperGraph.show_topo_graph(W,real_manifolds)

            # show clustering results
            plot_tools.autoPlot(X2,Y2)

            
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

            plot_tools.PaperGraph.show_graph(ConnectMat,manifolds,fileroot=figroot+'/{}.png',fileidx=20000000)
            plot_tools.PaperGraph.show_graph(W,manifolds,fileroot=figroot+'/{}.png',fileidx=30000000)
            plot_tools.PaperGraph.show_point_with_clusters(Sets,manifolds,fileroot=figroot+'/{}.png',fileidx=40000000,title='k:{} search_n:{} ratio:{}'.format(k,search_n,ratio))
            ploter.SaveGIF(mp4name,fps=fps)

        return X2,Y2,X_extend[:,-2],W