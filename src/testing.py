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

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

nokeepfiles=True
ElRes=32

data = pd.read_csv('ellipsoids_example4.csv')

fig = plt.figure()

Ellipsoids={}
for i in range(len(data)):
    
    # instantiate #####################################
    #Ellipsoids[i]=Icosahedron(2**i,data['description'][i])
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

ax = fig.add_subplot(111, projection='3d')

#maxbound=max(data['A'][0],data['B'][0],data['C'][0])
maxbound=max(1.5,1.5,1.5)
minbound = -1.0*maxbound
ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])

X=Y=Z={}
#ax.scatter(x,y,z,marker='o')
c=np.array(['r','g','b','w','r','g','b'])
ls=np.array(['-','-','-','-','-','-','-'])
lw=np.array(['0','0','0','0.5','0.5','0.5','0.5'])
for i in range(len(data)):
    print(i)
    for d in range(10):
        if d<11:
            for t in range((ElRes)*(ElRes)*2):
                start=3*2*d*(ElRes)*(ElRes)+3*t
            #for t in range((2**i)*(2**i)*2):
            #   start=3*2*d*(2**i)*(2**i)+3*t
                end=start+2
                print("t = ",t,"start = ",start,"end = ",end)
                X=Ellipsoids[i].TP[start:end+1,0]
                Y=Ellipsoids[i].TP[start:end+1,1]
                Z=Ellipsoids[i].TP[start:end+1,2]
            
                X=np.append(X,Ellipsoids[i].TP[start,0])
                Y=np.append(Y,Ellipsoids[i].TP[start,1])
                Z=np.append(Z,Ellipsoids[i].TP[start,2])
                
                #ax.plot(X,Y,Z,color=c[i],linestyle=ls[i],linewidth=lw[i])
                ax.plot_trisurf(X,Y,Z,color=c[i],linestyle=ls[i],linewidth=lw[i],alpha=0.95)
                
  
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
  
ax.auto_scale_xyz([minbound, maxbound], [minbound, maxbound], [minbound, maxbound])
plt.show()
