# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 22:03:48 2015

@author: sacim
"""

from collada import *
import numpy as np

#vertices
v=np.array([[ 00., 00., 50.],
            [-50.,-50.,-50.],
            [ 50.,-50.,-50.],
            [ 50., 50.,-50.],
            [-50., 50.,-50.]])
v.reshape((1,15))


# normals
n=np.array([np.cross(v[1]-v[0],v[2]-v[1]),
            np.cross(v[1]-v[0],v[2]-v[1]),
            np.cross(v[1]-v[0],v[2]-v[1]),
            np.cross(v[2]-v[0],v[3]-v[2]),
            np.cross(v[2]-v[0],v[3]-v[2]),
            np.cross(v[2]-v[0],v[3]-v[2]),
            np.cross(v[3]-v[0],v[4]-v[3]),
            np.cross(v[3]-v[0],v[4]-v[3]),
            np.cross(v[3]-v[0],v[4]-v[3]),
            np.cross(v[4]-v[0],v[1]-v[4]),
            np.cross(v[4]-v[0],v[1]-v[4]),
            np.cross(v[4]-v[0],v[1]-v[4]),
            np.cross(v[4]-v[1],v[3]-v[4]),
            np.cross(v[4]-v[1],v[3]-v[4]),
            np.cross(v[4]-v[1],v[3]-v[4]),
            np.cross(v[4]-v[2],v[3]-v[4])])
#n=np.array([[  0., 10., -5.],
#            [  0., 10., -5.],
#            [  0., 10., -5.],
#            [-10.,  0., -5.],
#            [-10.,  0., -5.],
#            [-10.,  0., -5.],
#            [  0.,-10., -5.],
#            [  0.,-10., -5.],
#            [  0.,-10., -5.],
#            [ 10.,  0., -5.],
#            [ 10.,  0., -5.],
#            [ 10.,  0., -5.],
#            [  0.,  0., 10.],
#            [  0.,  0., 10.],
#            [  0.,  0., 10.],
#            [  0.,  0., 10.],
#            [  0.,  0., 10.],
#            [  0.,  0., 10.]])

n.reshape((1,48))

n=-1.*n

mesh = Collada()
effect = material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
mat = material.Material("material0", "mymaterial", effect)
mesh.effects.append(effect)
mesh.materials.append(mat)

# What are these!?
vert_src = source.FloatSource("triverts-array", numpy.array(v), ('X', 'Y', 'Z'))
normal_src = source.FloatSource("trinormals-array", numpy.array(n), ('X', 'Y', 'Z'))


geom = geometry.Geometry(mesh, "geometry0", "mytri", [vert_src, normal_src])


input_list = source.InputList()
input_list.addInput(0, 'VERTEX', "#triverts-array")
input_list.addInput(1, 'NORMAL', "#trinormals-array")

indices = numpy.array([0,0,1,1,2,2,0,3,2,4,3,5,0,6,3,7,4,8,0,9,4,10,1,11,1,12,4,13,2,14,2,14,4,13,3,15])
#indices = numpy.array([0,1,4,1,2,4,2,3,4,3,0,4,2,1,3,0,3,1])

triset = geom.createTriangleSet(indices, input_list, "materialref")
geom.primitives.append(triset)
mesh.geometries.append(geom)

matnode = scene.MaterialNode("materialref", mat, inputs=[])
geomnode = scene.GeometryNode(geom, [matnode])
node = scene.Node("node0", children=[geomnode])

myscene = scene.Scene("myscene", [node])
mesh.scenes.append(myscene)
mesh.scene = myscene

mesh.write('/home/sacim/tri.dae')











