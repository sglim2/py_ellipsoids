#!/bin/python
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
import math
from mayavi import mlab
import icosahedron as ico
from collada import *

def create_ellipsoid_prametric(a,b,c,ngrid=24):
    u = np.linspace(0, 2*np.pi, num=ngrid, endpoint=True)
    v = np.linspace(0, np.pi, num=ngrid, endpoint=True)

    # Cartesian representation of data
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    return (x,y,z)
    
    
    
def rotate_point_about_xaxis(alpha, point):
    """Returns a point rotated by alpha radians about the x axis.
    
    """
    Rx=np.array([[             1,                0,               0 ],
                 [             0,  math.cos(alpha), math.sin(alpha) ],
                 [             0, -math.sin(alpha), math.cos(alpha) ]])
    
    return np.dot(Rx,np.transpose(point));


def rotate_point_about_yaxis(beta, point):
    """Returns a point rotated by beta radians about the y axis.
    
    """
    Ry=np.array([[  np.cos(beta),              0, -np.sin(beta) ],
                 [             0,              1,             0 ],
                 [  np.sin(beta),              0,  np.cos(beta) ]])
    
    return np.dot(Ry,np.transpose(point));
    
    
def rotate_point_about_zaxis(gamma, point):
    """Returns a point rotated by gamma radians about the z axis.
    
    """
    Rz=np.array([[ np.cos(gamma),  np.sin(gamma),             0 ],
                 [-np.sin(gamma),  np.cos(gamma),             0 ],
                 [             0,              0,             1 ]])
    
    return np.dot(Rz,np.transpose(point));
    
    
def rotate_point_about_u(theta, point, u):
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



# Google-Earth has a limits of 64k vertices
(TP,NP,indices,NT) = ico.create_icosahedron_optimized(8)

########################################
########################################
    
# build coefficients ==================
#    x**2      y**2      z**2
#    ----   +  ----   +  ----    =   1
#    a**2      b**2      c**2
#======================================
coeffs=np.array((25.,50.,100.))

# Define rotations
alpha=np.pi/5  # 90 degrees 
beta =np.pi/4  # 90 degrees 
gamma=np.pi/3  # 90 degrees 
    
# Scale to ellipsoid
for i in range(3*NT):
    TP[i,0]=coeffs[0]*TP[i,0]
    TP[i,1]=coeffs[1]*TP[i,1]
    TP[i,2]=coeffs[2]*TP[i,2]

# Rotate about x-axis, rotated points now stored in [xyz]r1
# Perform rotation on individual points about X-axis
u=np.array([1.,0.,0.])
for i in range(3*NT):
    (TP[i,0],TP[i,1],TP[i,2]) = rotate_point_about_u(-1.*alpha,[TP[i,0],TP[i,1],TP[i,2]],u)


# Rotate about object's x-axis, rotated points now stored in [xyz]r2
# Perform rotation on individual points about u
u=np.array([math.cos(alpha),(-1)*math.sin(alpha),0.])
for i in range(3*NT):
    (TP[i,0],TP[i,1],TP[i,2]) = rotate_point_about_u(-1.*beta,[TP[i,0],TP[i,1],TP[i,2]],u);

       
# Rotate about object's x-axis, rotated points now stored in [xyz]r2
# Perform rotation on individual points about u
u=np.array([math.sin(alpha)*math.cos(beta),
            math.cos(alpha)*math.cos(beta),
           (-1.)*math.sin(beta)*math.cos(alpha)])
for i in range(3*NT):
    (TP[i,0],TP[i,1],TP[i,2]) = rotate_point_about_u(1.*gamma,[TP[i,0],TP[i,1],TP[i,2]],u);
                
     
     
mesh = Collada()
effect = material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
mat = material.Material("material0", "mymaterial", effect)
mesh.effects.append(effect)
mesh.materials.append(mat)
    
    
vert_src = source.FloatSource("triverts-array", np.array(TP), ('X', 'Y', 'Z'))
normal_src = source.FloatSource("trinormals-array", np.array(NP), ('X', 'Y', 'Z'))

geom = geometry.Geometry(mesh, "geometry0", "mytri", [vert_src, normal_src])


input_list = source.InputList()
input_list.addInput(0, 'VERTEX', "#triverts-array")
input_list.addInput(1, 'NORMAL', "#trinormals-array")


triset = geom.createTriangleSet(indices, input_list, "materialref")
geom.primitives.append(triset)
mesh.geometries.append(geom)

matnode = scene.MaterialNode("materialref", mat, inputs=[])
geomnode = scene.GeometryNode(geom, [matnode])
node = scene.Node("node0", children=[geomnode])

myscene = scene.Scene("myscene", [node])
mesh.scenes.append(myscene)
mesh.scene = myscene

mesh.write('/home/sacim/ellipsoid.dae')



# Create a KML document
from simplekml import Kml, Model, AltitudeMode, Orientation, Scale

kml = Kml()
kml.document.name = "Ellipsoids"

mod = kml.newmodel(altitudemode=AltitudeMode.relativetoground,
                   #address=r'/home/sacim/ellipsoid.dae',
                   location="<longitude>-3.0722</longitude><latitude>51.5333</latitude><altitude>15.0</altitude>",
                   visibility=1,
                   )
mod.link.href="files/ellipsoid.dae"
kml.addfile("/home/sacim/ellipsoid.dae")

print kml.kml()
kml.savekmz("/home/sacim/testSimpleKML.kmz")
