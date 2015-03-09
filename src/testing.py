# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:31:04 2015

@author: sacim
"""

import math
import numpy as np
import pandas as pd
from collada import *
from simplekml import Kml, Model, AltitudeMode, Orientation, Scale
from icosahedron import Icosahedron
import argparse, os, csv

nokeepfiles=True
ElRes=8

data = pd.read_csv('ellipsoids_example3.csv')


Ellipsoids={}
for i in range(len(data)):
    
    # instantiate #####################################
    Ellipsoids[i]=Icosahedron(ElRes,data['description'][i])
    
    # re-shape ########################################
    ax=([data['A'][i],data['B'][i],data['C'][i]])
    ax.sort(key=float,reverse=True)
    Ellipsoids[i].stretch(ax[0],ax[1],ax[2])
    
    #Define Rotations ################################
    alpha=data['alpha'][i]*math.pi/180.
    beta =data['beta'][i] *math.pi/180.
    gamma=data['gamma'][i]*math.pi/180.
  
    # Rotate ellipsoid to match user-defined orientation 
    Ellipsoids[i].rotate_AlphaBetaGamma(alpha,beta,gamma) 
      
    # Rotate ellipsoid to match google-earth coordinates
    #Ellipsoids[i].rotate_eulerXY(math.pi,math.pi/2.)
                
    
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

i=0
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
maxbound=max(data['A'][i],data['B'][i],data['C'][i])
minbound = -1.0*maxbound
ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])
ax.scatter(Ellipsoids[i].TP[0::3], 
           Ellipsoids[i].TP[1::3], 
           Ellipsoids[i].TP[2::3],
           marker='.')
    
fig.show()