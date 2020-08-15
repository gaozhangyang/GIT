from pathlib import Path
import os
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import signal
import abc
from multiprocessing import Pool,Manager,Process
import numpy as np

colors = {-1:'gray'}
hashColor=lambda y:[colors[one] for one in y]

def autoPlot(data,y=None,continues=False):
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
        colors.update({i:'C{}'.format(i) for i in range(max(y)+1)})
        if data.shape[1]==2:
            ax = plt.subplot(111)
            ax.scatter(data[:,0], data[:,1],c=hashColor(y))

        if data.shape[1]==3:
            ax = plt.subplot(111, projection='3d')
            ax.scatter(data[:,0], data[:,1], data[:,2],c=hashColor(y))

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
    

    def run(self,pnum,tasks):
        signal.signal(signal.SIGTERM, term)

        ###############################多线程处理##########################
        Process_state=Manager().dict({str(i):True for i in range(pnum)})
        idx=0
        pid=1
        while idx<len(tasks):
            self.callback(tasks[idx],pid,Process_state)
            #查询是否有可用线程
            for pid in range(pnum):
                if Process_state[str(pid)]==True:
                    Process_state[str(pid)]=False #占用当前线程
                    p=Process(target=self.callback,args=(tasks[idx],pid,Process_state))
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
    def __init__(self,root):
        super(Visualization,self).__init__(root)
        self.color_list=['C{}'.format(i) for i in range(1,21)]
        self.Cnum=len(self.color_list)
    
    def callback(self,task,pid,Process_state):
        X, (root,B_mask) ,idx=task
        
        colors=np.array(['gray']*root.shape[0])
        for rt in list(set(root)):
            selected=np.nonzero(root==rt)[0].tolist()
            colors[selected]=self.color_list[int(rt%self.Cnum)]

        colors[np.nonzero(B_mask)]='gray'
        plt.figure(figsize=(4,4))
        plt.scatter(X[:,0],X[:,1],c=colors)
        plt.title('{}'.format(idx))
        plt.axis('equal')
        
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
    def show_topo_graph(self,W,manifolds,X_extend,fileroot=None,fileidx=1000):
        plt.figure(figsize=(4, 4))
        tmp_G=nx.from_numpy_matrix(W)
        centers=[]
        for i in range(len(manifolds)):
            centers.append(np.mean(X_extend[manifolds[i].pID,:-2],axis=0))

        pos={i:(centers[i][0],centers[i][1]) for i in range(len(manifolds)) }
        nx.draw_networkx(tmp_G,pos)
        edge_labels = nx.get_edge_attributes(tmp_G, 'weight')
        edge_labels={key:'{:.2f}'.format(val) for key,val in edge_labels.items()}
        nx.draw_networkx_edge_labels(tmp_G, pos, edge_labels=edge_labels)
        plt.axis('equal')
        if fileroot:
            for i in range(5):
                plt.savefig(Path(fileroot.format(fileidx+i)))
    

#     @classmethod
#     def show_point_with_clusters(self,sets,manifolds,fileroot=None,fileidx=1000,title=None):
#         colors={}
#         colors.update({i:'C{}'.format(i) for i in range(len(sets))})
#         plt.figure(figsize=(4, 4))
#         for i_cls,oneset in enumerate(sets):
#             for i_M in list(oneset):
#                 plt.scatter(manifolds[i_M].points[:,0],manifolds[i_M].points[:,1],c=colors[i_cls])
# #                 plt.text(manifolds[i_M].center[0],manifolds[i_M].center[1],'{:.0f}'.format(i_M))
#         plt.axis('equal')
#         if title:
#             plt.title(title)
#         if fileroot:
#             for i in range(15):
#                 plt.savefig(Path(fileroot.format(fileidx+i)))
#         plt.show()
    
