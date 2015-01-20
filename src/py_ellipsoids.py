#!/bin/python
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
from mayavi import mlab

def create_ellipsoid_prametric(a,b,c,ngrid=24):
    u = np.linspace(0, 2*np.pi, num=ngrid, endpoint=True)
    v = np.linspace(0, np.pi, num=ngrid, endpoint=True)

    # Cartesian representation of data
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    return (x,y,z)
    
def create_icosahedron_vertices():
    
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
    A[6][0] = v;  A[6][1] = 0; A[6][2] =-u; 
    A[7][0] = 0;  A[7][1] = u; A[7][2] =-v;
    A[8][0] =-u;  A[8][1] = v; A[8][2] = 0;
    A[9][0] =-u;  A[9][1] =-v; A[9][2] = 0;
    A[10][0]= 0;  A[10][1]=-u; A[10][2]=-v;  
    A[11][0]=-v;  A[11][1]= 0; A[11][2]=-u;
    
    return A
    
def create_ellipsoid_icosahedron(a,b,c,ngrid=24):
    #create unit sphere
  
    for id in range(1):
    if id<5:
	# Northern Hemisphere
	index = idx(0, 0, 0);
	xn[index] = Ad[0][0];
	yn[index] = Ad[0][1];
	zn[index] = Ad[0][2];
	// mt,0
	index = idx(0, mt, 0);
	xn[index] = Ad[id+1][0];
	yn[index] = Ad[id+1][1];
	zn[index] = Ad[id+1][2]; 
	// 0,mt
	index = idx(0, 0, mt);
	if (id == 0) {
	    xn[index] = Ad[id+5][0];
	    yn[index] = Ad[id+5][1];
	    zn[index] = Ad[id+5][2]; 
	}else{
	    xn[index] = Ad[id][0];
	    yn[index] = Ad[id][1];
	    zn[index] = Ad[id][2]; 
	}
	// mt,mt
	index = idx(0, mt, mt);
	xn[index] = Ad[id+6][0];
	yn[index] = Ad[id+6][1];
	zn[index] = Ad[id+6][2]; 
	
	// Southern Hemisphere
    }else{
	// South Pole
	index = idx(0, 0, 0);
	xn[index] = Ad[11][0];
	yn[index] = Ad[11][1];
	zn[index] = Ad[11][2];
	// mt,0
	index = idx(0, mt, 0);
	if (id == 9) {
	    xn[index] = Ad[id-3][0];
	    yn[index] = Ad[id-3][1];
	    zn[index] = Ad[id-3][2]; 
	}else{
	    xn[index] = Ad[id+2][0];
	    yn[index] = Ad[id+2][1];
	    zn[index] = Ad[id+2][2]; 
	}
	// 0,mt
	index = idx(0, 0, mt);
	xn[index] = Ad[id+1][0];
	yn[index] = Ad[id+1][1];
	zn[index] = Ad[id+1][2]; 
	// mt,mt
	index = idx(0, mt, mt);
	xn[index] = Ad[id-4][0];
	yn[index] = Ad[id-4][1];
	zn[index] = Ad[id-4][2]; 
    }
    x=np.zeros(ngrid)
    y=np.zeros()
    z=np.zeros()
  
    return
    
    
    
def rotate_about_xaxis(alpha, point):
    """Returns a point rotated by alpha radians about the x axis.
    
    """
    Rx=np.array([[             1,                0,               0 ],
                 [             0,  math.cos(alpha), math.sin(alpha) ],
                 [             0, -math.sin(alpha), math.cos(alpha) ]])
    
    return np.dot(Rx,np.transpose(point));


def rotate_about_yaxis(beta, point):
    """Returns a point rotated by beta radians about the y axis.
    
    """
    Ry=np.array([[  np.cos(beta),              0, -np.sin(beta) ],
                 [             0,              1,             0 ],
                 [  np.sin(beta),              0,  np.cos(beta) ]])
    
    return np.dot(Ry,np.transpose(point));
    
    
def rotate_about_zaxis(gamma, point):
    """Returns a point rotated by gamma radians about the z axis.
    
    """
    Rz=np.array([[ np.cos(gamma),  np.sin(gamma),             0 ],
                 [-np.sin(gamma),  np.cos(gamma),             0 ],
                 [             0,              0,             1 ]])
    
    return np.dot(Rz,np.transpose(point));
    
    
def rotate_about_u(theta, point, u):
    """Returns a point rotated by angle about vector u.
    
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


########################################
########################################
    
# build coefficients ==================
#    x**2      y**2      z**2
#    ----   +  ----   +  ----    =   1
#    a**2      b**2      c**2
#======================================
#coefs = (10., 3., 1.)

ngrid=5
(x,y,z)=create_ellipsoid_prametric(1.,1.,1.,ngrid)

A=create_ellipsoid_icosahedron(10.,3.,1.,7)

xn=A[:,0]
yn=A[:,1]
zn=A[:,2]



# Set of all x,y,z after rotations
xr1=np.zeros((ngrid,ngrid))
yr1=np.zeros((ngrid,ngrid))
zr1=np.zeros((ngrid,ngrid))
xr2=np.zeros((ngrid,ngrid))
yr2=np.zeros((ngrid,ngrid))
zr2=np.zeros((ngrid,ngrid))
xr3=np.zeros((ngrid,ngrid))
yr3=np.zeros((ngrid,ngrid))
zr3=np.zeros((ngrid,ngrid))


#alpha=np.pi/6  # 30 degrees 
#beta =np.pi/4   # 45 degrees 
#gamma=np.pi/6  # 30 degrees 

alpha=np.pi/2  # 90 degrees 
beta =np.pi/2  # 90 degrees 
gamma=np.pi/2  # 90 degrees 
    
# Rotate about x-axis, rotated points now stored in [xyz]r1
# Perform rotation on individual points about X-axis
u=np.array([1.,0.,0.])
for i in range(ngrid):
  for j in range(ngrid):
    (xr1[i,j],yr1[i,j],zr1[i,j]) = rotate_about_u(-1.*alpha,([x[i,j],y[i,j],z[i,j]]),u);


# Rotate about object's x-axis, rotated points now stored in [xyz]r2
# Perform rotation on individual points about u
u=np.array([math.cos(alpha),(-1)*math.sin(alpha),0.])
for i in range(ngrid):
  for j in range(ngrid):
    (xr2[i,j],yr2[i,j],zr2[i,j]) = rotate_about_u(-1.*beta,([xr1[i,j],yr1[i,j],zr1[i,j]]),u);

       
# Rotate about object's x-axis, rotated points now stored in [xyz]r2
# Perform rotation on individual points about u
u=np.array([math.sin(alpha)*math.cos(beta),
            math.cos(alpha)*math.cos(beta),
           (-1.)*math.sin(beta)*math.cos(alpha)])
for i in range(ngrid):
  for j in range(ngrid):
    (xr3[i,j],yr3[i,j],zr3[i,j]) = rotate_about_u(gamma,([xr2[i,j],yr2[i,j],zr2[i,j]]),u);
                 
       
#m = mlab.mesh(x  , y  , z  , color=(0.0, 0.0, 1.0))  # blue
#n = mlab.mesh(xr1, yr1, zr1, color=(0.0, 1.0, 0.0))  # green
#o = mlab.mesh(xr2, yr2, zr2, color=(1.0, 0.0, 0.0))  # red
mlab.mesh(x, y, z, color=(1.0, 1.0, 0.0))  # yellow
mlab.mesh(x, y, z, representation='wireframe', color=(0, 0, 0))
mlab.show()

mlab.scatter3


# convert to collada
from collada import *



