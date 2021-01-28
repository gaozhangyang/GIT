from pathlib import Path
import os
import imageio
import matplotlib.pyplot as plt
from matplotlib.pyplot import cla
import networkx as nx
import signal
import abc
from multiprocessing import Pool,Manager,Process
import numpy as np


def autoPlot(data,y=None,continues=False,seed=2020,area=None):
    plt.figure(figsize=(4, 4))
    if type(y)==np.ndarray:
        y=y.reshape(-1)

    if y is None:
        ax = plt.subplot(111)
        ax.scatter(data[:,0], data[:,1])
        plt.axis('equal')
        plt.show()
        return

    if continues:
        if data.shape[1]==2:
            ax = plt.subplot(111)
            ax.scatter(data[:,0], data[:,1],c=y)

        if data.shape[1]==3:
            ax = plt.subplot(111, projection='3d')
            ax.scatter(data[:,0], data[:,1], data[:,2],c=y)
            
        plt.axis('equal')
        plt.show()
    
    if not continues:
        colors=['C{}'.format(i) for i in range(max(y)+1)]
        random.seed(seed)
        random.shuffle(colors)
        colors={i:colors[i] for i in range(max(y)+1)}
        colors[-1]='gray'
        hashColor=lambda y:[colors[one] for one in y]
        noise_mask=(y==-1)
        

        if data.shape[1]==2:
            ax = plt.subplot(111)
            if area is None:
                ax.scatter(data[~noise_mask][:,0], data[~noise_mask][:,1],c=hashColor(y[~noise_mask]))
                ax.scatter(data[noise_mask][:,0], data[noise_mask][:,1],c=hashColor(y[noise_mask]),alpha=0.1)
            else:
                ax.scatter(data[~noise_mask][:,0], data[~noise_mask][:,1],c=hashColor(y[~noise_mask]), s=25 * area[~noise_mask].astype(np.float64) ** 3, alpha=0.5)
                ax.scatter(data[noise_mask][:,0], data[noise_mask][:,1],c=hashColor(y[noise_mask]), s=area[noise_mask].astype(np.float64), alpha=0.1)

        if data.shape[1]==3:
            ax = plt.subplot(111, projection='3d')
            ax.scatter(data[~noise_mask][:,0], data[~noise_mask][:,1], data[~noise_mask][:,2],c=hashColor(y[~noise_mask]))
            ax.scatter(data[noise_mask][:,0], data[noise_mask][:,1], data[noise_mask][:,2],c=hashColor(y[noise_mask]),alpha=0.1)

        plt.axis('equal')
        plt.show()

    
import plotly.graph_objects as go
import plotly.express as px
named_colors=px.colors.qualitative.Light24

def autoPlotly(X,labels):
    
    hovertexts=[str(label) for label in labels]
    
    fig = go.Figure(data=go.Scatter(
        x = X[:,0], # non-uniform distribution
        y = X[:,1], # zoom to see more points at the center
        mode='markers',
        marker=dict(
            color=[named_colors[int(i)%24] for i in labels],
            line_width=1
        ),
        hoverinfo = 'text',
        hovertext = hovertexts,
    ))
    
    fig.update_layout(
        width = 500,
        height = 500,
        title = "",
        xaxis = dict(
          range=[-2,2],  # sets the range of xaxis
          constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
        ),
        yaxis = dict(
          scaleanchor = "x",
          scaleratio = 1,
        ),
    )

    fig.show()


processes=[]

def term(sig_num, addtion):
    print('terminate process {}'.format(os.getpid()))
    try:
        print('the processes is {}'.format(processes) )
        for p in processes:
            print('process {} terminate'.format(p.pid))
            p.terminate()
            # os.kill(p.pid, signal.SIGKILL)
    except Exception as e:
        print(str(e))

class GIFPloter():
    def __init__(self,root):
        self.root=root

    @abc.abstractmethod
    def callback(self,task,pid,Process_state):
        # TODO 此处增加画图操作
        Process_state[str(pid)]=True
    

    def run(self,pnum,tasks,Y):
        signal.signal(signal.SIGTERM, term)

        ###############################多线程处理##########################
        Process_state=Manager().dict({str(i):True for i in range(pnum)})
        idx=0
        pid=1
        while idx<len(tasks):
            self.callback(tasks[idx],Y,pid,Process_state)
            #查询是否有可用线程
            for pid in range(pnum):
                if Process_state[str(pid)]==True:
                    Process_state[str(pid)]=False #占用当前线程
                    p=Process(target=self.callback,args=(tasks[idx],Y,pid,Process_state))
                    print( '{}'.format(idx) )
                    idx+=1
                    p.start()
                    processes.append(p)
                    break

        for p in processes:
            p.join()


    def SaveGIF(self,name,fps=1):
        path = self.root
        gif_images_path = os.listdir(path+'/')

        gif_images_path=[img for img in gif_images_path if img[-4:]=='.png']
        gif_images_path=sorted(gif_images_path,key=lambda x:int(x[:-4]))
        gif_images = []
        for i, path_ in enumerate(gif_images_path):
            print(path_)
            if '.png' in path_:
                if i % 1 == 0:
                    gif_images.append(imageio.imread(path+'/'+path_))
                    
        imageio.mimsave(path+'/'+"{}.gif".format(name), gif_images, fps=fps)



class Visualization(GIFPloter):
    def __init__(self,root,seed=2020):
        super(Visualization,self).__init__(root)
        self.color_list=['C{}'.format(i) for i in range(1,21)]
        random.seed(seed)
        random.shuffle(self.color_list)
        self.Cnum=len(self.color_list)
    
    def callback(self,task,Y,pid,Process_state):
        X, root, B_mask ,idx=task
        root=Y
        
        colors=np.array(['gray']*root.shape[0])
        for rt in list(set(root)):
            selected=np.nonzero(root==rt)[0].tolist()
            colors[selected]=self.color_list[int(rt%self.Cnum)]

        colors[np.nonzero(B_mask)]='gray'
        plt.figure(figsize=(4,4))
        B_mask=(colors=='gray')
        plt.scatter(X[~B_mask,0],X[~B_mask,1],c=colors[~B_mask])
        plt.scatter(X[B_mask,0],X[B_mask,1],c=colors[B_mask],alpha=0.1)
        plt.title('{}'.format(idx))
        plt.axis('equal')
        plt.xticks([])
        plt.yticks([])
        
        path=Path(self.root)
        path.mkdir(exist_ok=True,parents=True)
        plt.savefig(path/('{}.png'.format(idx)))
        plt.close()

        Process_state[str(pid)]=True


import random
class PaperGraph:
    def __init__(self) -> None:
        pass

    @classmethod
    def show_manifolds(self,manifolds,X_extend,seed=2020):
        NC=len(manifolds)
        color_list=['gray']
        color_add=['C{}'.format(i) for i in range(NC)]
        color_add.remove('C7')
        random.seed(seed)
        random.shuffle(color_add)
        color_list=color_list+color_add
        
        fig=plt.figure(figsize=(4, 4))
        for i in range(NC):
            points=X_extend[manifolds[i].pID,:-2]
            centers=np.mean(points,axis=0)
            plt.scatter(points[:,0],points[:,1],c=color_list[i])
            plt.text(centers[0],centers[1],str(i))
        
        _=plt.axis('equal')
    
    
    @classmethod
    def show_clusters(self,X,Y,fileroot=None,fileidx=1000,title=None):
        C=int(np.max(Y))+1
        colors={-1:'gray'}
        colors.update({i:'C{}'.format(i) for i in range(C)})
        plt.figure(figsize=(4, 4))
        for c in range(-1,C):
            mask= (Y==c)
            if np.sum(mask) >0 :
                plt.scatter(X[mask][:,0],X[mask][:,1],c=colors[c])
        plt.axis('equal')

        if title:
            plt.title(title)
        if fileroot:
            for i in range(15):
                plt.savefig(Path(fileroot.format(fileidx+i)))
        # plt.show()

    @classmethod
    def show_local_clusters(self,X,V,seed=2020):
        Ks=list(V.keys())
        Vs=list(V.values())

        colors=['C{}'.format(i) for i in range(len(Ks))]
        random.seed(seed)
        random.shuffle(colors)
        plt.figure(figsize=(4, 4))
        for c in range(len(Vs)):
            if Ks[c]==-1:
                plt.scatter(X[Vs[c],0],X[Vs[c],1],c='gray',alpha=0.3)
            else:
                plt.scatter(X[Vs[c],0],X[Vs[c],1],c=colors[c])
                # plt.text(X[Vs[c],0].mean(),X[Vs[c],1].mean(),'{:.0f}'.format(c))
        plt.axis('equal')
        plt.show()
    
    @classmethod
    def show_topo_graph(self,V,E,X=None,fileroot=None,fileidx=1000):
        plt.figure(figsize=(4, 4))
        G = nx.Graph()
        G.add_nodes_from(V.keys())
        for e in E:
            if E[e]>0:
                G.add_edge(e[0],e[1],weight=E[e])
        if -1 in G.nodes():
            G.remove_node(-1)
        pos={}
        if X is None:
            pos = nx.kamada_kawai_layout(G)
        else:
            for i in V.keys():
                pos[i]=X[V[i]].mean(axis=0)

        nx.draw_networkx(G,pos,with_labels=False,node_size =50)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        edge_labels={key:'{:.2f}'.format(val) for key,val in edge_labels.items()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
        plt.axis('equal')

        if fileroot:
            for i in range(5):
                plt.savefig(Path(fileroot.format(fileidx+i)))
    
