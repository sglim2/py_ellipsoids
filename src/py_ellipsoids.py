#!/bin/python
import math
import numpy as np
from collada import *
from icosahedron import Icosahedron

def parse_config():
    configfile=open('ellipsoids.cfg', 'r')
    lines =configfile.readlines()
    configfile.close()

    data={}
    header=lines[0]
    params=header.split()

    name={}
    key=0
    for line in lines[1:]:
        if line == '\n':
            continue
        words = line.split()
        name[key] = words[0]
        values = words[1:]
        data[name[key]]={}
        for p, v in zip(params, values):
            if v != 'n/a':
                data[name[key]][p] = float(v)
        key += 1
    return name,data




(names,data)=parse_config()
# Google-Earth has a limits of 64k vertices

Ellipsoids={}
for i in range(len(names)):
    # instantiate
    Ellipsoids[i]=Icosahedron(16,names[i])
    
    # re-shape
    Ellipsoids[i].stretch(data[names[i]]['A'],
                          data[names[i]]['B'],
                          data[names[i]]['C'])
    
    #Define Rotations
    alpha=data[names[i]]['alpha'] 
    beta =data[names[i]]['beta'] 
    gamma=data[names[i]]['gamma']

    u=np.array([1.,0.,0.])
    Ellipsoids[i].rotate_about_u(alpha,u)
    
    
    u=np.array([math.cos(alpha),(-1)*math.sin(alpha),0.])
    Ellipsoids[i].rotate_about_u(beta,u)
    
    u=np.array([math.sin(alpha)*math.cos(beta),
                math.cos(alpha)*math.cos(beta),
                (-1.)*math.sin(beta)*math.cos(alpha)])
    Ellipsoids[i].rotate_about_u(gamma,u)


    # Create Collada Object and writer to tmp file
    mesh = Collada()
    effect = material.Effect("effect0", [], "phong", diffuse=(1,0,0), specular=(0,1,0))
    mat = material.Material("material0", "mymaterial", effect)
    mesh.effects.append(effect)
    mesh.materials.append(mat)
      
    vert_src = source.FloatSource("triverts-array", np.array(Ellipsoids[i].TP), ('X', 'Y', 'Z'))
    normal_src = source.FloatSource("trinormals-array", np.array(Ellipsoids[i].NP), ('X', 'Y', 'Z'))

    geom = geometry.Geometry(mesh, "geometry0", "mytri", [vert_src, normal_src])

    input_list = source.InputList()
    input_list.addInput(0, 'VERTEX', "#triverts-array")
    input_list.addInput(1, 'NORMAL', "#trinormals-array")

    triset = geom.createTriangleSet(Ellipsoids[i].indices, input_list, "materialref")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    matnode = scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = scene.GeometryNode(geom, [matnode])
    node = scene.Node("node0", children=[geomnode])

    myscene = scene.Scene("myscene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene

    mesh.write('./'+Ellipsoids[i].name+'.dae')






# Create a KML document
from simplekml import Kml, Model, AltitudeMode, Orientation, Scale

kml = Kml()
kml.document.name = "Ellipsoids"

                   
for i in range(len(names)):
    mod = kml.newmodel(altitudemode=AltitudeMode.relativetoground,
                       location='<longitude>'+repr(data[names[i]]['lon'])+'</longitude>'+
                                '<latitude>'+repr(data[names[i]]['lat'])+'</latitude>'+
                                '<altitude>'+repr(data[names[i]]['alt'])+'</altitude>',
                       visibility=1,
                       )
    mod.link.href=('files/'+Ellipsoids[i].name+'.dae')
    kml.addfile('./'+Ellipsoids[i].name+'.dae')

#print kml.kml()
kml.savekmz("./Ellipsoids.kmz")




########################
########################
########################



