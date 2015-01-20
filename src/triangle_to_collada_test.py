# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 22:03:48 2015

@author: sacim
"""

from collada import *
import numpy as np

#vertices
v=np.array([[-1.,-1.,-1.],
            [ 1.,-1.,-1.],
            [ 1., 1.,-1.],
            [-1., 1.,-1.],
            [ 0., 0., 1.]])
            # normals
n=np.array([[0.,1.,0.],
            [0.,1.,0.],
            [0.,1.,0.],
            [0.,1.,0.],
            [0.,1.,0.]])

print t

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

indices = numpy.array([0,1,2])

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











