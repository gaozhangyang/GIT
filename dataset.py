from sklearn.datasets import make_blobs
from sklearn.datasets import make_circles
from sklearn.datasets import make_moons
from sklearn import cluster, datasets, mixture
import matplotlib.pyplot as plt

import numpy as np
import random as rd
import torch

colors = {-1:'gray',0:'red', 1:'blue',2:'green'}
hashColor=lambda y:[colors[one] for one in y]

def SetSeed(seed):
    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    rd.seed(SEED)
    np.random.seed(SEED)

def load_data(name,n_samples,noiseX=0.05,noiseY=0.1,SP=0.1,Seed=2020,plot=True):
    SetSeed(Seed)
    if name=='blobs':
        # generate 2d classification dataset
        X, y = make_blobs(n_samples=n_samples, centers=2, n_features=2)
    
    if name=='circles':
        # generate 2d classification dataset
        X, y = make_circles(n_samples=n_samples, noise=noiseX, factor=0.6)
    
    if name=='moons':
        X, y = make_moons(n_samples=n_samples, noise=noiseX)
    
    if name=='varied':
        X,y = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5])
    
    if name=='aniso':
        X, y = datasets.make_blobs(n_samples=n_samples)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)

    if plot:
        # scatter plot, dots colored by class value
        plt.figure(figsize=(4, 4))
        plt.scatter(X[:,0],X[:,1], label=y,c=hashColor(y))
        _=plt.axis('equal')
        plt.show()
    
    known_mask=np.random.rand(X.shape[0])<=SP
    Y=np.eye(np.max(y)+1)[y.reshape(-1)]

    Y_init=Y.copy()
    Y_init[~known_mask]=0
    Y_init+=np.random.rand(Y.shape[0],Y.shape[1])*noiseY

    return X, Y, known_mask,Y_init