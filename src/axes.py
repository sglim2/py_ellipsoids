# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 09:41:29 2015

@author: sacim
"""


import numpy as np
import math

#=============================================================================
class Axes:
    def __init__(self, name='ellipsoid'):
        self.name    = name                 # A short name of our Axes
        self.TP      = np.zeros((6*3*3,3))   # Vertices of triangles
        self.NP      = np.zeros((6*3*3,3))   # Normal at each triangle vertex
        self.indices = np.zeros(6*3*3*2,dtype=int)  # vertex and normal locations (for use with pycollada)
        
        self.create_axes()

#=============================================================================
    def create_axes(self):
        """
        Creates an axes object for converting to a COLLADA object.
        """
        self.name=self.name+'_axes'

        A=0.02
        B=0.8
        C=1.0
        D=0.1
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
        TP3=rotate_point_about_zaxis(math.pi/2,TP1[:])
        TP4=rotate_point_about_zaxis(math.pi/2,TP2[:])
        TP5=rotate_point_about_yaxis(-math.pi/2,TP1[:])
        TP6=rotate_point_about_yaxis(-math.pi/2,TP2[:])
        self.TP=np.concatenate((TP1.reshape(3*len(TP1)),
                                TP2.reshape(3*len(TP2)),
                                TP3.reshape(3*len(TP3)),
                                TP4.reshape(3*len(TP4)),
                                TP5.reshape(3*len(TP5)),
                                TP6.reshape(3*len(TP6))),axis=0).reshape((6*9,3))
        self.NP[::3]=self.NP[1::3]=self.NP[2::3]=np.cross(self.TP[::3]-self.TP[1::3],self.TP[2::3]-self.TP[1::3])
        
        for i in range(len(self.TP)):
            self.indices[2*i+0]=self.indices[2*i+1]=i
                         
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
             
#///////////////////////////////////////////////////////////////////////////// 
def colours(colour):
    """
    Defines the rgb vaules of named colours
    """
    colours = {'white'  : np.array([1.00,1.00,1.00]),
               'grey'   : np.array([0.50,0.50,0.50]),
               'black'  : np.array([0.00,0.00,0.00])}
   
    return colours[colour]
    
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
