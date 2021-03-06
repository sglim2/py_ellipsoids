# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 09:41:29 2015

@author: sacim
"""


import numpy as np
import math

#=============================================================================
class Icosahedron:
    def __init__(self, ngrid, name='ellipsoid'):
        self.name    = name                 # A short name of our icosahedron
        self.ngrid   = ngrid                # grid size for each domain
        self.NT      = 10*2*ngrid**2        # Number of triangles
        self.TP      = np.zeros((3*self.NT,3))   # Vertices of triangles
        self.NP      = np.zeros((3*self.NT,3))   # Normal at each triangle vertex
        self.indices = np.zeros(3*2*self.NT,dtype=int) # vertex and normal locations (for use with pycollada)
        
        self.create_icosahedron()
        
    def create_icosahedron(self):
        """
        Creates an icosahedron
        
        A regular icosahedron is initialized with 12 vertices defined by array
        A (see code). This defines a regular icosahedron with vertices 
        normalised to  unity. Each vertex is then rotated by Beta (see code)
        about the y-axis. This gives us a regular icosahedron with 'poles' 
        along the z-axis.
        
        From this point the icosahedron is split into 10 domains, consisting
        of 2 joined equilateral trianlges, 5 with a vertex position at the 
        north pole, and five with a vertex at the south pole.
        
        Within each domain, ngrid-1 geodesic points are calculated between 
        adjacent vertices, giving ngrid+1 points along each domain edge.
        
        The remaining geodesic points are then calculated by walking through
        suitable pairs of domain edge points, and calculating the suitable 
        number of points between these pairs.
        
        Each domain is split into triangle points and stored in a numpy array
        TP. TP triplets consist of x, y, and z coordinates of the calculated 
        triangles.
        
        The numpy array NP holds the normal of each vertex of a TP triangle.
        
        the numpy array indices holds the indices of the triangles and their
        associated normals in a format suitable for processing by the  
        pycollada package.
        
        """ 
        ngrid=self.ngrid
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
        Ad=np.zeros((12,3))
        for i in range(12):
            Ad[i,:]=rotate_point_about_yaxis(Beta,A[i,:])
    
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
        

        
        TP1=np.zeros((self.NT,3)) # triangle point 1 x,y,z positions
        TP2=np.zeros((self.NT,3)) #        "       2      "
        TP3=np.zeros((self.NT,3)) #        "       3      "
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
                     

        
        self.TP[0::3,:]=TP1[:,:]
        self.TP[1::3,:]=TP2[:,:]
        self.TP[2::3,:]=TP3[:,:]

        NTP1=np.zeros((self.NT,3))
        NTP2=np.zeros((self.NT,3))
        NTP3=np.zeros((self.NT,3))

        for t in range(self.NT):
            NTP1[t,:] = np.cross(TP1[t,:]-TP2[t,:],TP3[t,:]-TP2[t,:])
            NTP3[t,:] = NTP2[t,:] = NTP1[t,:]

        self.NP[0::3,:]=NTP1[:,:]
        self.NP[1::3,:]=NTP2[:,:]
        self.NP[2::3,:]=NTP3[:,:]

        self.NP=-1.*self.NP

        for i in range(3*self.NT):
            self.indices[2*i+0]=self.indices[2*i+1]=i

#============================================================================= 
    def get_centre(self):
        """
        Finds the centre of the ellipsoid, simply by averaging two points 
        lying on the (perhaps rotated) x semi-axis.
        """
        centre=np.zeros(3)
        centre[0]=0.5*(self.TP[0,0]+self.TP[-2,0])
        centre[1]=0.5*(self.TP[0,1]+self.TP[-2,1])
        centre[2]=0.5*(self.TP[0,2]+self.TP[-2,2])    
        
        return centre
#============================================================================= 
    def translate(self,x,y,z):
        """
        Shifts the origin by x,y,z
    
        """ 
        self.TP[:,0]=self.TP[:,0]+x
        self.TP[:,1]=self.TP[:,1]+y
        self.TP[:,2]=self.TP[:,2]+z
        
#============================================================================= 
    def translateTo(self,x,y,z):
        """
        Shifts the origin to x,y,z
    
        """ 
        centre=self.get_centre()
        dx=x-centre[0]
        dy=y-centre[1]
        dz=z-centre[2]
        
        self.translate(dx,dy,dz)
        
#============================================================================= 
    def stretch(self,A,B,C):
        """
        Stretches the icosahedron by a factor of A,B,C along x,y,z axes respectively
        
        Before scaling, the icosahedron is centred at the origin, and returned
        to its original position afterwards.
        
        """     
        # translate to origin before stretch
        centre=self.get_centre()
        self.translateTo(0.,0.,0.)
        
        self.TP[:,0]=A*self.TP[:,0]
        self.TP[:,1]=B*self.TP[:,1]
        self.TP[:,2]=C*self.TP[:,2]
            
        self.NP[:,0]=A*self.NP[:,0]
        self.NP[:,1]=B*self.NP[:,1]
        self.NP[:,2]=C*self.NP[:,2]
        
        # return to original position
        self.translateTo(centre[0],centre[1],centre[2])
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
        Rotate icosahedron by theta radians about vector u
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
        Rotate icosahedron by alpha radians about X, then by beta radians about
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
        Rotate icosahedron by alpha radians about Y, then by beta radians about
        X. Rotation matrices about Y,X-axes are pre-multiplied to give a new
        R matrix:
          R = XY
        Rotation is performed by R acting on each point on the icosahedron.
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
        Rotate icosahedron by alpha radians about Y, then by beta radians about
        X. Rotation matrices about Y,X-axes are pre-mutliplied to give a new
        R matrix:
          R = XY
        Rotation is performed by R acting on each point on the icosahedron.
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
#=============================================================================
 
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
    
    return np.dot(Rx,np.transpose(point));
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
    
    return np.dot(Ry,np.transpose(point));
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
    
    return np.dot(Rz,np.transpose(point));
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
#============================================================================= 
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
#============================================================================= 

         
