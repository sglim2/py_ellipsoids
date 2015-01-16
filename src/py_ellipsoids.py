#!/bin/python
import numpy as np
import matplotlib.pyplot as plt
import math


def rotate_about_xaxis(angle, point):
    """Returns a point rotated by angle radians about the x axis.
    
    """
    Rx=np.array([[             1,                0,               0 ],
                 [             0,  math.cos(alpha), math.sin(alpha) ],
                 [             0, -math.sin(alpha), math.cos(alpha) ]])
    
    return np.dot(Rx,np.transpose(point));


def rotate_about_yaxis(angle, point):
    """Returns a point rotated by angle radians about the y axis.
    
    """
    Ry=np.array([[  np.cos(beta),              0, -np.sin(beta) ],
                 [             0,              1,             0 ],
                 [  np.sin(beta),              0,  np.cos(beta) ]])
    
    return np.dot(Ry,np.transpose(point));
    
    
def rotate_about_zaxis(angle, point):
    """Returns a point rotated by angle radians about the z axis.
    
    """
    Rz=np.array([[ np.cos(gamma),  np.sin(gamma),             0 ],
                 [-np.sin(gamma),  np.cos(gamma),             0 ],
                 [             0,              0,             1 ]])
    
    return np.dot(Rz,np.transpose(point));
    
    
def rotate_about_xyz(angleX, angleY, angleZ, point):
    """Returns a point rotated by angleX, angleY, and angleZ radians about the
       x-, y-, and z-axis respectively.
    
    """
    Rz=np.array([[ np.cos(gamma),  np.sin(gamma),             0 ],
                 [-np.sin(gamma),  np.cos(gamma),             0 ],
                 [             0,              0,             1 ]])
    
    return np.dot(Rz,np.transpose(point));


########################################
########################################
    
# build coefficients ==================
#    x**2      y**2      z**2
#    ----   +  ----   +  ----    =   1
#    a**2      b**2      c**2
#======================================
coefs = (5., 3., 1.)

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


alpha=np.pi/6
beta=np.pi/4
gamma=np.pi/2
    
# Rotate about x-axis, rotated points now stored in [xyz]r1
# Perform rotation on individual points about X-axis
for i in range(ngrid):
  for j in range(ngrid):
    (xr1[i,j],yr1[i,j],zr1[i,j]) = rotate_about_xaxis(alpha,([x[i,j],y[i,j],z[i,j]]));

                        
# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_wireframe(x, y, z, rstride=4, cstride=4, color='k', alpha=0.25)
ax.plot_wireframe(
                  xr1, yr1, zr1, rstride=8, cstride=8, color='k', alpha=0.25)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
#plt.show()
