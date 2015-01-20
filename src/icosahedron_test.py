# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 20:49:38 2015

@author: sacim
"""

import numpy as np
import math

def midpt(P1, P2):
    P=np.array((P1[0]+P2[0],P1[1]+P2[1],P1[2]+P2[2]))
        
    return P/math.sqrt(np.sum(P**2))


a=1.
tau = 0.5*(math.sqrt(5.)+1.)
rho=tau-1.
u=a/(math.sqrt(1.+rho**2.))
v=rho*u

A =np.zeros((12,3))
        
A[0][0] = v;  A[0][1] = 0; A[0][2] = u; 
A[1][0] = u;  A[1][1] = v; A[1][2] = 0; 
A[2][0] = 0;  A[2][1] = u; A[2][2] = v;   
A[3][0] =-v;  A[3][1] = 0; A[3][2] = u;  
A[4][0] = 0;  A[4][1] =-u; A[4][2] = v;  
A[5][0] = u;  A[5][1] =-v; A[5][2] = 0;
A[6][0] = 0;  A[6][1] = u; A[6][2] =-v;
A[7][0] =-u;  A[7][1] = v; A[7][2] = 0;
A[8][0] =-u;  A[8][1] =-v; A[8][2] = 0;
A[9][0] = 0;  A[9][1] =-u; A[9][2] =-v;
A[10][0]= v;  A[10][1]= 0; A[10][2]=-u;   
A[11][0]=-v;  A[11][1]= 0; A[11][2]=-u;
    
    
Beta= math.atan(v/u);
Ry=np.zeros((3,3))
Ry[0][0] = math.cos(Beta);
Ry[0][1] = 0.;
Ry[0][2] = -math.sin(Beta);
Ry[1][0] = 0.;
Ry[1][1] = 1.;
Ry[1][2] = 0.;
Ry[2][0] = math.sin(Beta);
Ry[2][1] = 0.;
Ry[2][2] = math.cos(Beta);

Ad=np.dot(Ry,np.transpose(A))

Ad=np.zeros((12,3))
for i in range(12):
    Ad[i,:]=np.dot(Ry,np.transpose(A[i,:]))

ngrid=4
xn=np.zeros(10*(ngrid+1)**2)
yn=np.zeros(10*(ngrid+1)**2)
zn=np.zeros(10*(ngrid+1)**2)

for id in range(10):
     if id<5:
         # Northern Hemisphere
         # 0,0
         xn[id*(ngrid+1)**2+0] = Ad[0][0]
         yn[id*(ngrid+1)**2+0] = Ad[0][1]
         zn[id*(ngrid+1)**2+0] = Ad[0][2]
         # mt, 0
         xn[id*(ngrid+1)**2+ngrid] = Ad[id+1][0]
         yn[id*(ngrid+1)**2+ngrid] = Ad[id+1][1]
         zn[id*(ngrid+1)**2+ngrid] = Ad[id+1][2]
         # 0,mt
         if id==4:
             xn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[1][0]
             yn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[1][1]
             zn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[1][2]
         else:
             xn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[id+2][0]
             yn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[id+2][1]
             zn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[id+2][2]
             # mt,mt
         xn[id*(ngrid+1)**2+(ngrid+1)**2-1] = Ad[id+6][0]
         yn[id*(ngrid+1)**2+(ngrid+1)**2-1] = Ad[id+6][1]
         zn[id*(ngrid+1)**2+(ngrid+1)**2-1] = Ad[id+6][2]
	# Southern Hemisphere
     else:
         # South Pole
         xn[id*(ngrid+1)**2+0] = Ad[id-4][0];
         yn[id*(ngrid+1)**2+0] = Ad[id-4][1];
         zn[id*(ngrid+1)**2+0] = Ad[id-4][2];
         # mt,0
         if id==5:
             xn[id*(ngrid+1)**2+ngrid] = Ad[10][0];
             yn[id*(ngrid+1)**2+ngrid] = Ad[10][1];
             zn[id*(ngrid+1)**2+ngrid] = Ad[10][2]; 
         else:
             xn[id*(ngrid+1)**2+ngrid] = Ad[id][0];
             yn[id*(ngrid+1)**2+ngrid] = Ad[id][1];
             zn[id*(ngrid+1)**2+ngrid] = Ad[id][2]; 
	    # 0,mt
         xn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[id+1][0];
         yn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[id+1][1];
         zn[id*(ngrid+1)**2+(ngrid+1)*ngrid] = Ad[id+1][2]; 
         # mt,mt
         xn[id*(ngrid+1)**2+(ngrid+1)**2-1] = Ad[11][0];
         yn[id*(ngrid+1)**2+(ngrid+1)**2-1] = Ad[11][1];
         zn[id*(ngrid+1)**2+(ngrid+1)**2-1] = Ad[11][2]; 
    
    
    
for k in range(1,int(1.45*math.log(ngrid))):
    
     m  = int(2**k+0.1)
     l  = ngrid/m
     l2 = l/2

     # rows of diamond--
     for j1 in range(m+1):
         for j2 in range(m):
             i1 = j1*l;
             i2 = j2*l + l2;
             index = id*(ngrid+1)**2 + ngrid*i2 + i1
             index1= id*(ngrid+1)**2 + ngrid*(i2-l2) + i1
             index2= id*(ngrid+1)**2 + ngrid*(i2+l2) + i1
             T=midpt((xn[index1],yn[index1],zn[index1]),(xn[index2],yn[index2],zn[index2]));
             xn[index]=T[0]
             xn[index]=T[1]
             xn[index]=T[2]
	  


verts = [zip(xn,yn,zn)]
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xn, yn, zn)
plt.show()
    
    
    
    
    
    
    
    