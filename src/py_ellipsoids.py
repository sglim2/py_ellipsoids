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
            
            

def create_ellipsoid_parametric(a,b,c,ngrid=24):
    u = np.linspace(0, 2*np.pi, num=ngrid, endpoint=True)
    v = np.linspace(0, np.pi, num=ngrid, endpoint=True)

    # Cartesian representation of data
    x = a * np.outer(np.cos(u), np.sin(v))
    y = b * np.outer(np.sin(u), np.sin(v))
    z = c * np.outer(np.ones_like(u), np.cos(v))
    
    return (x,y,z)
    
def merge_collada_files(list_fpath_inputs, fpath_output):
    '''
    code lifted from:
    https://groups.google.com/forum/#!topic/pycollada/SEC_TQbpRgQ
    '''
    list_collada_objects = []
    for fpath_input in list_fpath_inputs:
        list_collada_objects.append(collada.Collada(fpath_input))
    merged_collada_object = merge_collada_objects(list_collada_objects)
    merged_collada_object.write(fpath_output)

def merge_collada_objects(list_collada_objects):
    '''
    code lifted from:
    https://groups.google.com/forum/#!topic/pycollada/SEC_TQbpRgQ
    ''' 
    merged_collada_object = collada.Collada()  
    
    if len(list_collada_objects) == 0:
        return merged_collada_object
   
    merged_collada_object.assetInfo = list_collada_objects[0].assetInfo

    list_nodes_of_scene = []
    for mesh in list_collada_objects:
        merged_collada_object.effects.extend(mesh.effects)
        merged_collada_object.materials.extend(mesh.materials)
        merged_collada_object.geometries.extend(mesh.geometries)
       
        for scene in mesh.scenes:
            list_nodes_of_scene.extend(scene.nodes)         
       
    myscene = collada.scene.Scene("myscene", list_nodes_of_scene)
    merged_collada_object.scenes.append(myscene)
    merged_collada_object.scene = myscene       
           
    return merged_collada_object
 
 



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

    mesh.write('/home/sacim/'+Ellipsoids[i].name+'.dae')






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
    kml.addfile('/home/sacim/'+Ellipsoids[i].name+'.dae')

print kml.kml()
kml.savekmz("/home/sacim/Ellipsoids.kmz")




########################
########################
########################



