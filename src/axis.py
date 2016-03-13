# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 09:41:29 2015

@author: sacim
"""


import numpy as np
import math

#=============================================================================
class Axis:
    def __init__(self, name='axis'):
        self.name    = name                  # A short name of our Axes
        self.TP      = np.zeros((2*3*3,3))   # Vertices of triangles
        self.NP      = np.zeros((2*3*3,3))   # Normal at each triangle vertex
        self.indices = np.zeros(2*3*3*2,dtype=int)  # vertex and normal locations (for use with pycollada)
        self.origin  = np.array([0.,0.,0.])  # The origin of the Axes. This point must be translated along
                                             # with any translate function of the Axes data
        
        self.create_axes()

#=============================================================================
    def create_axes(self):
        """
        Creates an axes object for converting to a COLLADA object.
        """
        #self.name=self.name+'_axes'

        A=0.005
        B=0.95
        C=1.0
        D=0.02
        TP1=np.array([[0. , -A , 0.],
                      [B  , -A , 0.],
                      [0. ,  A , 0.],
                      [0. ,  A , 0.],
                      [B  , -A , 0.],
                      [B  ,  A , 0.],
                      [B  , -D , 0.],
                      [C  ,  0., 0.],
                      [B  ,  D , 0.]])
        #TP2=np.zeros((5,3))
        TP2=rotate_point_about_xaxis(math.pi/2,TP1[:])
        self.TP=np.concatenate((TP1.reshape(3*len(TP1)),
                                TP2.reshape(3*len(TP2))),axis=0).reshape((2*9,3))
        self.NP[::3]=self.NP[1::3]=self.NP[2::3]=np.cross(self.TP[::3]-self.TP[1::3],self.TP[2::3]-self.TP[1::3])
        
        for i in range(len(self.TP)):
            self.indices[2*i+0]=self.indices[2*i+1]=i

#============================================================================= 
    def get_centre(self):
        """
        Finds the centre of the Axes.
        """
        
        return self.origin
        
#============================================================================= 
    def translate(self,x,y,z):
        """
        Shifts the origin by x,y,z
    
        """ 
        self.TP[:,0]=self.TP[:,0]+x
        self.TP[:,1]=self.TP[:,1]+y
        self.TP[:,2]=self.TP[:,2]+z
        self.origin[0]+=x
        self.origin[1]+=y
        self.origin[2]+=z
        
#============================================================================= 
    def translateTo(self,x,y,z):
        """
        Shifts the origin to x,y,z
    
        """ 
        origin=self.origin
        dx=x-origin[0]
        dy=y-origin[1]
        dz=z-origin[2]
        
        self.translate(dx,dy,dz)
        
#============================================================================= 
    def stretch(self,A,B,C):
        """
        Stretches the axes by a factor of A,B,C along x,y,z axes respectively
        
        Before scaling, the axes is centred at the origin, and returned
        to its original position afterwards.
        
        """     
        # translate to origin before stretch
        self.translateTo(0.,0.,0.)
        
        self.TP[:,0]=A*self.TP[:,0]
        self.TP[:,1]=B*self.TP[:,1]
        self.TP[:,2]=C*self.TP[:,2]
            
        self.NP[:,0]=A*self.NP[:,0]
        self.NP[:,1]=B*self.NP[:,1]
        self.NP[:,2]=C*self.NP[:,2]
        
        # return to original position
        self.translateTo(self.origin[0],self.origin[1],self.origin[2])                      
#============================================================================= 
    def rotate_about_xaxis(self, alpha):
        """
        Rotates the icosahedron by beta radians about the x axis. Rotation is
        anti-clockwise as you look down the axis (from a positive viewpoint) 
        towards the origin.   
        """
        alpha=-alpha
        Rx=np.array([[             1,                0,               0 ],
                     [             0,  math.cos(alpha), math.sin(alpha) ],
                     [             0, -math.sin(alpha), math.cos(alpha) ]])
        
        for i in range(len(self.TP)):
            point = [self.TP[i,0],self.TP[i,1],self.TP[i,2]]
            (self.TP[i,0],
             self.TP[i,1],
             self.TP[i,2]) = np.dot(Rx,np.transpose(point))
             
            npoint = [self.NP[i,0],self.NP[i,1],self.NP[i,2]] 
            (self.NP[i,0],
             self.NP[i,1],
             self.NP[i,2]) = np.dot(Rx,np.transpose(npoint))  
#=============================================================================    
    def rotate_about_yaxis(self, beta):
        """
        Rotates the icosahedron by beta radians about the y axis. Rotation is
        anti-clockwise as you look down the axis (from a positive viewpoint) 
        towards the origin.
        """
        beta=-beta 
        Ry=np.array([[  np.cos(beta),              0, -np.sin(beta) ],
                     [             0,              1,             0 ],
                     [  np.sin(beta),              0,  np.cos(beta) ]])
                     
        for i in range(len(self.TP)):
            point = [self.TP[i,0],self.TP[i,1],self.TP[i,2]]
            (self.TP[i,0],
             self.TP[i,1],
             self.TP[i,2]) = np.dot(Ry,np.transpose(point))
             
            npoint = [self.NP[i,0],self.NP[i,1],self.NP[i,2]] 
            (self.NP[i,0],
             self.NP[i,1],
             self.NP[i,2]) = np.dot(Ry,np.transpose(npoint))
                         
#=============================================================================
    def rotate_about_zaxis(self, gamma):
        """
        Rotates the icosahedron by beta radians about the z axis. Rotation is
        anti-clockwise as you look down the axis (from a positive viewpoint) 
        towards the origin.    
        """
        gamma=-gamma
        Rz=np.array([[ np.cos(gamma),  np.sin(gamma),             0 ],
                     [-np.sin(gamma),  np.cos(gamma),             0 ],
                     [             0,              0,             1 ]])
        
        for i in range(len(self.TP)):
            point = [self.TP[i,0],self.TP[i,1],self.TP[i,2]]
            (self.TP[i,0],
             self.TP[i,1],
             self.TP[i,2]) = np.dot(Rz,np.transpose(point))
             
            npoint = [self.NP[i,0],self.NP[i,1],self.NP[i,2]] 
            (self.NP[i,0],
             self.NP[i,1],
             self.NP[i,2]) = np.dot(Rz,np.transpose(npoint))
             
#=============================================================================
    def rotate_about_u(self,theta,u):
        """
        Rotate Axes by theta radians about vector u
        u must be normalized.
        """      
    
        u = normalize(u)
        
        for i in range(len(self.TP)):
            (self.TP[i,0],
             self.TP[i,1],
             self.TP[i,2]) = rotate_point_about_u(theta,[self.TP[i,0],
                                                         self.TP[i,1],
                                                         self.TP[i,2]],
                                                        u)
            (self.NP[i,0],
             self.NP[i,1],
             self.NP[i,2]) = rotate_point_about_u(theta,[self.NP[i,0],
                                                         self.NP[i,1],
                                                         self.NP[i,2]],
                                                        u)
                                                                        
#============================================================================= 
    def rotate_eulerYX(self,alpha,beta):
        """
        Rotate Axes by alpha radians about X, then by beta radians about
        Y. Rotation matrices about X,Y-axes are pre-multiplied to give a new
        R matrix:
          R = YX
        Rotation is performed by R acting on each point on the icosahedron.
        Rotation is anti-clockwise as you look down the axis (from a positive 
        viewpoint) towards the origin.
        """ 
        alpha=-alpha
        beta=-beta
        
        sa=math.sin(alpha)
        sb=math.sin(beta)
        ca=math.cos(alpha)
        cb=math.cos(beta)
       
        R=np.array([[ cb ,  sa*sb ,  -ca*sb ],
                    [  0 ,   ca   ,    sa   ],
                    [ sb , -cb*sa ,   ca*cb ]])
    
        for i in range(len(self.TP)):
            point = [self.TP[i,0],self.TP[i,1],self.TP[i,2]]
            (self.TP[i,0],
             self.TP[i,1],
             self.TP[i,2]) = np.dot(R,np.transpose(point))
             
            npoint = [self.NP[i,0],self.NP[i,1],self.NP[i,2]] 
            (self.NP[i,0],
             self.NP[i,1],
             self.NP[i,2]) = np.dot(R,np.transpose(npoint))
             
#============================================================================= 
    def rotate_eulerXY(self,alpha,beta):
        """
        Rotate Axes by alpha radians about Y, then by beta radians about
        X. Rotation matrices about Y,X-axes are pre-multiplied to give a new
        R matrix:
          R = XY
        Rotation is performed by R acting on each point on the Axes.
        """ 
        alpha=-alpha
        beta=-beta
        
        sa=math.sin(alpha)
        sb=math.sin(beta)
        ca=math.cos(alpha)
        cb=math.cos(beta)
       
        R=np.array([[ ca    ,   0   ,  -sa   ],
                    [ sa*sb ,   cb  ,  ca*sb ],
                    [ cb*sa ,  -sb  ,  ca*cb ]])
    
        for i in range(len(self.TP)):
            point = [self.TP[i,0],self.TP[i,1],self.TP[i,2]]
            (self.TP[i,0],
             self.TP[i,1],
             self.TP[i,2]) = np.dot(R,np.transpose(point))
             
            npoint = [self.NP[i,0],self.NP[i,1],self.NP[i,2]] 
            (self.NP[i,0],
             self.NP[i,1],
             self.NP[i,2]) = np.dot(R,np.transpose(npoint))

#============================================================================= 
    def rotate_eulerZY(self,alpha,beta):
        """
        Rotate Axes by alpha radians about Y, then by beta radians about
        X. Rotation matrices about Y,X-axes are pre-multiplied to give a new
        R matrix:
          R = XY
        Rotation is performed by R acting on each point on the Axes.
        """ 
        alpha=-alpha
        beta=-beta
        
        sa=math.sin(alpha)
        sb=math.sin(beta)
        ca=math.cos(alpha)
        cb=math.cos(beta)
       
        R=np.array([[ ca*cb ,   sb  , -cb*sa ],
                    [-ca*sb ,   cb  ,  sa*sb ],
                    [  sa   ,   0   ,   ca   ]])
    
        for i in range(len(self.TP)):
            point = [self.TP[i,0],self.TP[i,1],self.TP[i,2]]
            (self.TP[i,0],
             self.TP[i,1],
             self.TP[i,2]) = np.dot(R,np.transpose(point))
             
            npoint = [self.NP[i,0],self.NP[i,1],self.NP[i,2]] 
            (self.NP[i,0],
             self.NP[i,1],
             self.NP[i,2]) = np.dot(R,np.transpose(npoint))
             
#============================================================================= 
    def rotate_AlphaBetaGamma(self,alpha,beta,gamma):
        """
        
        """ 
        self.rotate_eulerZY(alpha,beta)

        u=normalize(rotate_eulerZY(alpha,beta,[1,0,0]))
        self.rotate_about_u(gamma,u)     
#///////////////////////////////////////////////////////////////////////////// 
def normalize(P):
    """
    Nomalizes a point P.
    """ 
    return P/math.sqrt(np.sum(P**2))           

#///////////////////////////////////////////////////////////////////////////// 
def rotate_point_about_xaxis(alpha, point):
    """
    Returns a point rotated by alpha radians about the x axis. Rotation is
    anti-clockwise as you look down the axis (from a positive viewpoint) 
    towards the origin.
    
    """
    alpha=-alpha
    Rx=np.array([[             1,                0,               0 ],
                 [             0,  math.cos(alpha), math.sin(alpha) ],
                 [             0, -math.sin(alpha), math.cos(alpha) ]])
    
    return np.dot(Rx,np.transpose(point)).transpose();
#/////////////////////////////////////////////////////////////////////////////    
def rotate_point_about_yaxis(beta, point):
    """
    Returns a point rotated by beta radians about the y axis. Rotation is
    anti-clockwise as you look down the axis (from a positive viewpoint) 
    towards the origin.
    
    """
    beta=-beta 
    Ry=np.array([[  np.cos(beta),              0, -np.sin(beta) ],
                 [             0,              1,             0 ],
                 [  np.sin(beta),              0,  np.cos(beta) ]])
    
    return np.dot(Ry,np.transpose(point)).transpose();
#///////////////////////////////////////////////////////////////////////////// 
def rotate_point_about_zaxis(gamma, point):
    """
    Returns a point rotated by gamma radians about the z axis. Rotation is
    anti-clockwise as you look down the axis (from a positive viewpoint) 
    towards the origin.
    
    """
    gamma=-gamma       
    Rz=np.array([[ np.cos(gamma),  np.sin(gamma),             0 ],
                 [-np.sin(gamma),  np.cos(gamma),             0 ],
                 [             0,              0,             1 ]])
    
    return np.dot(Rz,np.transpose(point)).transpose();
    
#///////////////////////////////////////////////////////////////////////////// 
def rotate_point_about_u(theta, point, u):
    """
    Returns a point rotated by angle theta about vector u.
    u must be normalized.
    
    """ 
    u = normalize(u)
    
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
def rotate_eulerZY(alpha, beta, point):
    """
    Rotate a point by alpha radians about Y, then by beta radians about
    X. Rotation matrices about Y,X-axes are pre-multiplied to give a new
    R matrix:
      R = XY
    Rotation is performed by R acting on the point.    
    """ 
    alpha=-alpha
    beta=-beta
    
    sa=math.sin(alpha)
    sb=math.sin(beta)
    ca=math.cos(alpha)
    cb=math.cos(beta)
    
    R=np.array([[ ca*cb ,   sb  , -cb*sa ],
                [-ca*sb ,   cb  ,  sa*sb ],
                [  sa   ,   0   ,   ca   ]])

    return np.dot(R,np.transpose(point));
#/////////////////////////////////////////////////////////////////////////////
