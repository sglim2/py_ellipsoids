# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 08:52:03 2015

@author: Ian Merrick
"""

import numpy as np
import math

class Icosahedron:
    def __init__(self, ngrid):
        self.ngrid = ngrid

#///////////////////////////////////////////////////////////////////////////// 
def normalize(P):
    """
    Nomalizes a point
    
    """ 
    return P/math.sqrt(np.sum(P**2))
#///////////////////////////////////////////////////////////////////////////// 
def translate(I,x,y,z):
    """
    translate icosahedron I by x,y,z
    
    """ 
#///////////////////////////////////////////////////////////////////////////// 
def stretch(I,A,B,C):
    """
    stretch icosahedron I by a factor of A,B,C along x,y,z axes respectively
    
    """     
    for i in range(len(I)):
        I[i,0]=A*I[i,0]
        I[i,1]=B*I[i,1]
        I[i,2]=C*I[i,2]
    
    return I
#///////////////////////////////////////////////////////////////////////////// 
def rotate_point_about_xaxis(alpha, point):
    """Returns a point rotated by alpha radians about the x axis.
    
    """
    Rx=np.array([[             1,                0,               0 ],
                 [             0,  math.cos(alpha), math.sin(alpha) ],
                 [             0, -math.sin(alpha), math.cos(alpha) ]])
    
    return np.dot(Rx,np.transpose(point));
#/////////////////////////////////////////////////////////////////////////////    
def rotate_point_about_yaxis(beta, point):
    """Returns a point rotated by beta radians about the y axis.
    
    """
    Ry=np.array([[  np.cos(beta),              0, -np.sin(beta) ],
                 [             0,              1,             0 ],
                 [  np.sin(beta),              0,  np.cos(beta) ]])
    
    return np.dot(Ry,np.transpose(point));
#///////////////////////////////////////////////////////////////////////////// 
def rotate_point_about_zaxis(gamma, point):
    """Returns a point rotated by gamma radians about the z axis.
    
    """
    Rz=np.array([[ np.cos(gamma),  np.sin(gamma),             0 ],
                 [-np.sin(gamma),  np.cos(gamma),             0 ],
                 [             0,              0,             1 ]])
    
    return np.dot(Rz,np.transpose(point));
 #/////////////////////////////////////////////////////////////////////////////   
def rotate_point_about_u(theta, point, u):
    """Returns a point rotated by angle theta about vector u.
    
    """ 
    c=math.cos(theta)
    s=math.sin(theta)
    ux=u[0]
    uy=u[1]
    uz=u[2]
    ux2=u[0]*u[0]
    uy2=u[1]*u[1]
    uz2=u[2]*u[2]

    R=np.array([[ c+ux2*(1.-c)      ,  ux*uy*(1.-c) -uz*s, ux*uz*(1.-c)+uy*s ],
                [ uy*ux*(1-c) + uz*s,  c+uy2*(1.-c)      , uy*uz*(1.-c)-ux*s ],
                [ uz*ux*(1.-c)-uy*s ,  uz*uy*(1.-c)+ux*s , c+uz2*(1.-c)      ]])
    
    return np.dot(R,np.transpose(point));
#///////////////////////////////////////////////////////////////////////////// 
def rotate_about_xaxis(I,alpha):
    """
    Rotate all points of icosahedron I by alpha radians about the x axis
    
    """     
    for i in range(len(I)):
        (I[i,0],I[i,1],I[i,2]) = rotate_point_about_xaxis(-1.*alpha,[I[i,0],I[i,1],I[i,2]])
    
    return I
#///////////////////////////////////////////////////////////////////////////// 
def rotate_about_yaxis(I,beta):
    """
    Rotate all points of icosahedron I by beta radians about the y axis
    
    """     
    for i in range(len(I)):
        (I[i,0],I[i,1],I[i,2]) = rotate_point_about_yaxis(-1.*beta,[I[i,0],I[i,1],I[i,2]])
    
    return I
#///////////////////////////////////////////////////////////////////////////// 
def rotate_about_zaxis(I,gamma):
    """
    Rotate all points of icosahedron I by gamma radians about the z axis
    
    """     
    for i in range(len(I)):
        (I[i,0],I[i,1],I[i,2]) = rotate_point_about_zaxis(gamma,[I[i,0],I[i,1],I[i,2]])
    
    return I
#///////////////////////////////////////////////////////////////////////////// 
def rotate_about_u(I,theta,u):
    """
    Rotate all points of icosahedron I by theta radians about the vector u
    
    """     
    for i in range(len(I)):
        (I[i,0],I[i,1],I[i,2]) = rotate_point_about_u(theta,[I[i,0],I[i,1],I[i,2]],u)
    
    return I
#///////////////////////////////////////////////////////////////////////////// 
def create_icosahedron(ngrid=24):
    """
    Creates an icosahedron
    
    """ 
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

    xn=np.zeros(10*(ngrid+1)**2)
    yn=np.zeros(10*(ngrid+1)**2)
    zn=np.zeros(10*(ngrid+1)**2)

    # Define Domain corners
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
    
    
    
    # Define Domain points between corners
    for id in range(10):
        for k in range(1,ngrid):
            # upper left line
            index1 = id*(ngrid+1)**2 + 0
            index2 = id*(ngrid+1)**2 + ngrid
            index  = id*(ngrid+1)**2 + k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        
            # upper right line
            index1 = id*(ngrid+1)**2 + 0
            index2 = id*(ngrid+1)**2 + (ngrid+1)*ngrid
            index  = id*(ngrid+1)**2 + (ngrid+1)*k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        
	  
            # lower left line
            index1 = id*(ngrid+1)**2 + ngrid
            index2 = id*(ngrid+1)**2 + (ngrid+1)**2-1
            index  = id*(ngrid+1)**2 + (ngrid+1)*(k+1)-1
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
     
     
                # lower right line
            index1 = id*(ngrid+1)**2 + (ngrid+1)*ngrid
            index2 = id*(ngrid+1)**2 + (ngrid+1)**2-1
            index  = id*(ngrid+1)**2 + (ngrid+1)*ngrid+k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        

            # middle line
            index1 = id*(ngrid+1)**2 + ngrid
            index2 = id*(ngrid+1)**2 + (ngrid+1)*ngrid
            index  = id*(ngrid+1)**2 + (ngrid+1)*k+ngrid-k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        
        
        # Define Domain points all over 
        # top half triangle
        for k in range(2,ngrid):
            for l in range(1,k):
                index1 = id*(ngrid+1)**2 + k
                index2 = id*(ngrid+1)**2 + k*(ngrid+1)
                index  = id*(ngrid+1)**2 + k + l*(ngrid)
        
                xn[index] = xn[index1] + 1.*l*(xn[index2]-xn[index1])/k
                yn[index] = yn[index1] + 1.*l*(yn[index2]-yn[index1])/k
                zn[index] = zn[index1] + 1.*l*(zn[index2]-zn[index1])/k
                (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
            
            
        # bottom half triangle
        for k in range(2,ngrid):
            for l in range(1,ngrid+1-k):
                index1 = id*(ngrid+1)**2 + k*(ngrid+1) - 1
                index2 = id*(ngrid+1)**2 + (ngrid+1)*ngrid + k -1
                index  = id*(ngrid+1)**2 + k*(ngrid+1) -1 + l*(ngrid)
                
                xn[index] = xn[index1] + 1.*l*(xn[index2]-xn[index1])/(ngrid+1-k)
                yn[index] = yn[index1] + 1.*l*(yn[index2]-yn[index1])/(ngrid+1-k)
                zn[index] = zn[index1] + 1.*l*(zn[index2]-zn[index1])/(ngrid+1-k)
                (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        

    NT=10*2*ngrid**2
    TP1=np.zeros((NT,3)) # triangle point 1 x,y,z positions
    TP2=np.zeros((NT,3)) #        "       2      "
    TP3=np.zeros((NT,3)) #        "       3      "
    # Create triangles  
    for id in range(10):
        for k in range(ngrid):
            for l in range(ngrid):
            
                Tindex=id*2*ngrid*ngrid + k*2*ngrid + l
                
                index1=id*(ngrid+1)**2 + k*(ngrid+1) +l
                index2=id*(ngrid+1)**2 + k*(ngrid+1) +l+1
                index3=id*(ngrid+1)**2 + (k+1)*(ngrid+1) +l
            
                TP1[Tindex][0] = xn[index1]
                TP1[Tindex][1] = yn[index1]
                TP1[Tindex][2] = zn[index1]
                TP2[Tindex][0] = xn[index2]
                TP2[Tindex][1] = yn[index2]
                TP2[Tindex][2] = zn[index2]
                TP3[Tindex][0] = xn[index3]
                TP3[Tindex][1] = yn[index3]
                TP3[Tindex][2] = zn[index3]
                
            for l in range(0,ngrid):
            
                Tindex=id*2*ngrid*ngrid + k*2*ngrid + ngrid + l
                
                index1=id*(ngrid+1)**2 +     k*(ngrid+1) + l+1
                index2=id*(ngrid+1)**2 + (k+1)*(ngrid+1) + l+1
                index3=id*(ngrid+1)**2 + (k+1)*(ngrid+1) + l
                
                TP1[Tindex][0] = xn[index1]
                TP1[Tindex][1] = yn[index1]
                TP1[Tindex][2] = zn[index1]
                TP2[Tindex][0] = xn[index2]
                TP2[Tindex][1] = yn[index2]
                TP2[Tindex][2] = zn[index2]
                TP3[Tindex][0] = xn[index3]
                TP3[Tindex][1] = yn[index3]
                TP3[Tindex][2] = zn[index3]
                     

    TP=np.zeros((3*NT,3))
    TP[0::3,:]=TP1[:,:]
    TP[1::3,:]=TP2[:,:]
    TP[2::3,:]=TP3[:,:]

    NTP1=np.zeros((NT,3))
    NTP2=np.zeros((NT,3))
    NTP3=np.zeros((NT,3))

    for t in range(NT):
        NTP1[t,:] = np.cross(TP1[t,:]-TP2[t,:],TP3[t,:]-TP2[t,:])
        NTP3[t,:] = NTP2[t,:] = NTP1[t,:]

    NP=np.zeros((3*NT,3))
    NP[0::3,:]=NTP1[:,:]
    NP[1::3,:]=NTP2[:,:]
    NP[2::3,:]=NTP3[:,:]

    NP=-1.*NP

    indices=np.zeros(3*2*NT,dtype=int)
    for i in range(3*NT):
        indices[2*i+0]=indices[2*i+1]=i
        
    #indices=np.zeros((3*NT,2))
    #for i in range(NT):
    #    indices[3*i+0,0]=i
    #    indices[3*i+1,0]=i
    #    indices[3*i+2,0]=i
    #    indices[3*i+0,1]=i
    #    indices[3*i+1,1]=i
    #    indices[3*i+2,1]=i

    return TP,NP,indices,NT 
#/////////////////////////////////////////////////////////////////////////////  
def create_icosahedron_optimized(ngrid=24):
    """
    Creates an icosahderon
    
    """ 
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

    Ad=np.zeros((12,3))
    for i in range(12):
        Ad[i,:]=np.dot(Ry,np.transpose(A[i,:]))

    xn=np.zeros(10*(ngrid+1)**2)
    yn=np.zeros(10*(ngrid+1)**2)
    zn=np.zeros(10*(ngrid+1)**2)

    # Define Domain corners
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
    
    
    
    # Define Domain points between corners
    for id in range(10):
        for k in range(1,ngrid):
            # upper left line
            index1 = id*(ngrid+1)**2 + 0
            index2 = id*(ngrid+1)**2 + ngrid
            index  = id*(ngrid+1)**2 + k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        
            # upper right line
            index1 = id*(ngrid+1)**2 + 0
            index2 = id*(ngrid+1)**2 + (ngrid+1)*ngrid
            index  = id*(ngrid+1)**2 + (ngrid+1)*k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        
	  
            # lower left line
            index1 = id*(ngrid+1)**2 + ngrid
            index2 = id*(ngrid+1)**2 + (ngrid+1)**2-1
            index  = id*(ngrid+1)**2 + (ngrid+1)*(k+1)-1
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
     
     
                # lower right line
            index1 = id*(ngrid+1)**2 + (ngrid+1)*ngrid
            index2 = id*(ngrid+1)**2 + (ngrid+1)**2-1
            index  = id*(ngrid+1)**2 + (ngrid+1)*ngrid+k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        

            # middle line
            index1 = id*(ngrid+1)**2 + ngrid
            index2 = id*(ngrid+1)**2 + (ngrid+1)*ngrid
            index  = id*(ngrid+1)**2 + (ngrid+1)*k+ngrid-k
        
            xn[index] = xn[index1] + 1.*k*(xn[index2]-xn[index1])/ngrid
            yn[index] = yn[index1] + 1.*k*(yn[index2]-yn[index1])/ngrid
            zn[index] = zn[index1] + 1.*k*(zn[index2]-zn[index1])/ngrid
            (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        
        
        # Define Domain points all over 
        # top half triangle
        for k in range(2,ngrid):
            for l in range(1,k):
                index1 = id*(ngrid+1)**2 + k
                index2 = id*(ngrid+1)**2 + k*(ngrid+1)
                index  = id*(ngrid+1)**2 + k + l*(ngrid)
        
                xn[index] = xn[index1] + 1.*l*(xn[index2]-xn[index1])/k
                yn[index] = yn[index1] + 1.*l*(yn[index2]-yn[index1])/k
                zn[index] = zn[index1] + 1.*l*(zn[index2]-zn[index1])/k
                (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
            
            
        # bottom half triangle
        for k in range(2,ngrid):
            for l in range(1,ngrid+1-k):
                index1 = id*(ngrid+1)**2 + k*(ngrid+1) - 1
                index2 = id*(ngrid+1)**2 + (ngrid+1)*ngrid + k -1
                index  = id*(ngrid+1)**2 + k*(ngrid+1) -1 + l*(ngrid)
                
                xn[index] = xn[index1] + 1.*l*(xn[index2]-xn[index1])/(ngrid+1-k)
                yn[index] = yn[index1] + 1.*l*(yn[index2]-yn[index1])/(ngrid+1-k)
                zn[index] = zn[index1] + 1.*l*(zn[index2]-zn[index1])/(ngrid+1-k)
                (xn[index],yn[index],zn[index]) = normalize(np.array((xn[index],yn[index],zn[index])))
        

    NT=10*2*ngrid**2
    TP1=np.zeros((NT,3)) # triangle point 1 x,y,z positions
    TP2=np.zeros((NT,3)) #        "       2      "
    TP3=np.zeros((NT,3)) #        "       3      "
    # Create triangles  
    for id in range(10):
        for k in range(ngrid):
            for l in range(ngrid):
            
                Tindex=id*2*ngrid*ngrid + k*2*ngrid + l
                
                index1=id*(ngrid+1)**2 + k*(ngrid+1) +l
                index2=id*(ngrid+1)**2 + k*(ngrid+1) +l+1
                index3=id*(ngrid+1)**2 + (k+1)*(ngrid+1) +l
            
                TP1[Tindex][0] = xn[index1]
                TP1[Tindex][1] = yn[index1]
                TP1[Tindex][2] = zn[index1]
                TP2[Tindex][0] = xn[index2]
                TP2[Tindex][1] = yn[index2]
                TP2[Tindex][2] = zn[index2]
                TP3[Tindex][0] = xn[index3]
                TP3[Tindex][1] = yn[index3]
                TP3[Tindex][2] = zn[index3]
                
            for l in range(0,ngrid):
            
                Tindex=id*2*ngrid*ngrid + k*2*ngrid + ngrid + l
                
                index1=id*(ngrid+1)**2 +     k*(ngrid+1) + l+1
                index2=id*(ngrid+1)**2 + (k+1)*(ngrid+1) + l+1
                index3=id*(ngrid+1)**2 + (k+1)*(ngrid+1) + l
                
                TP1[Tindex][0] = xn[index1]
                TP1[Tindex][1] = yn[index1]
                TP1[Tindex][2] = zn[index1]
                TP2[Tindex][0] = xn[index2]
                TP2[Tindex][1] = yn[index2]
                TP2[Tindex][2] = zn[index2]
                TP3[Tindex][0] = xn[index3]
                TP3[Tindex][1] = yn[index3]
                TP3[Tindex][2] = zn[index3]
                     

    TP=np.zeros((3*NT,3))
    TP[0::3,:]=TP1[:,:]
    TP[1::3,:]=TP2[:,:]
    TP[2::3,:]=TP3[:,:]

    NTP1=np.zeros((NT,3))
    NTP2=np.zeros((NT,3))
    NTP3=np.zeros((NT,3))

    for t in range(NT):
        NTP1[t,:] = np.cross(TP1[t,:]-TP2[t,:],TP3[t,:]-TP2[t,:])
        NTP3[t,:] = NTP2[t,:] = NTP1[t,:]

    NP=np.zeros((3*NT,3))
    NP[0::3,:]=NTP1[:,:]
    NP[1::3,:]=NTP2[:,:]
    NP[2::3,:]=NTP3[:,:]

    NP=-1.*NP

    indices=np.zeros(3*2*NT,dtype=int)
    for i in range(3*NT):
        indices[2*i+0]=indices[2*i+1]=i
        
    #indices=np.zeros((3*NT,2))
    #for i in range(NT):
    #    indices[3*i+0,0]=i
    #    indices[3*i+1,0]=i
    #    indices[3*i+2,0]=i
    #    indices[3*i+0,1]=i
    #    indices[3*i+1,1]=i
    #    indices[3*i+2,1]=i

    return TP,NP,indices,NT  
#///////////////////////////////////////////////////////////////////////////// 