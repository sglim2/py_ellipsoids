#!/bin/python
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
from mayavi import mlab

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
coefs = (10., 3., 1.)

ngrid=120

# Set of all spherical angles:
u = np.linspace(0, 2*np.pi, num=ngrid, endpoint=True)
v = np.linspace(0, np.pi, num=ngrid, endpoint=True)

# Cartesian representation of data
x = coefs[0] * np.outer(np.cos(u), np.sin(v))
y = coefs[1] * np.outer(np.sin(u), np.sin(v))
z = coefs[2] * np.outer(np.ones_like(u), np.cos(v))

xr1=np.zeros((120,120))
yr1=np.zeros((120,120))
zr1=np.zeros((120,120))
xr2=np.zeros((120,120))
yr2=np.zeros((120,120))
zr2=np.zeros((120,120))
xr3=np.zeros((120,120))
yr3=np.zeros((120,120))
zr3=np.zeros((120,120))


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
p = mlab.mesh(xr3, yr3, zr3, color=(1.0, 1.0, 0.0))  # yellow
mlab.show()
                 
# plot with matplotlib
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(xr3, yr3, zr3, rstride=8, cstride=8, color='k', alpha=0.25)
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()
