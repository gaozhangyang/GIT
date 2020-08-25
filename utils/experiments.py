'''
experiments on toy datasets
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../utils/')
import imp
import plot_tools
imp.reload(plot_tools)
import pandas as pd
import networkx as nx

import api
imp.reload(api)
panda = pd.read_csv('./artificial_csv/circles_0.1_noise.csv', header=None)
X = panda.values[:,:2]

Y = api.DGSFC.fit(X,
                  K_d=30,
                  K_n=13,
                  lamda=0.3,
                  epsilon=0.6,
                  plot=True,
                  pnum=30,
                  mp4=False,
                  fps=4,
                  figroot='/usr/data/gzy/DGC_copy/ex1_toy/results/circles',
                  mp4name='circles'
                 )