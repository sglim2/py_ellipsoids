#!/bin/python
import math
import numpy as np
from collada import *
from simplekml import Kml, Model, AltitudeMode, Orientation, Scale
from icosahedron import Icosahedron
import argparse, os

# Parse input args ############
parser = argparse.ArgumentParser(description='Builds user-defined ellipsoids as Collada objects, and outputs a google kmz file containing these objects.')
#parser.add_argument('-i', '--input', nargs="?", help='The input file.')
parser.add_argument('input', help='The input file.')
parser.add_argument('output', help='The output file.')
parser.add_argument('-r', '-res', '--resolution', type=int, default=16, help='The resolution of the generated ellipsoids.')
parser.add_argument('--keep', action='store_true', help='Do not delete the intermediate collada files.')
args = parser.parse_args()

nokeepfiles=True
ElRes=16
if (args.keep):
   nokeepfiles=False
if (args.resolution):
   ElRes=args.resolution
###############################

def parse_config(inputfile):
    """
    Parses the config file. 
    
    The config file is a plain text file consisting of a table of values, with 
    headings. Each row defines an ellipsoid, and each column describes 
    properties of that ellipsoid. 
    The first column has no heading, and is the ellipsoid name or description
    (no spaces permitted). The remaining columns each have a heading, and may 
    be in any order. The headings are (with descriptions):
        A       (Ellipsoid semi-axis along the x-direction)
        B       (Ellipsoid semi-axis along the y-direction)
        C       (Ellipsoid semi-axis along the z-direction)
        lat     (Latitude position of the centre of the ellipsoid)
        lon     (Longitude position of the centre of the ellipsoid)
        alt     (Altitude position of the centre of the ellispsoid, relative to the ground)
        alpha   (Rotation, in radians, about the x-axis)
        beta    (Rotation, in radians, about the y-axis)
        gamma   (Rotation, in radians, about the z-axis)
        red     (red component of rgb colour: 0..1)
        green   (green compnent of rgb colour: 0..1)
        blue    (blue compnent of rgb colour: 0..1)
        
    Rotations of the ellipsoid are performed in the order: rotation about 
    x-axis, then y, then z.
    
        
    Returns:
      data:
      name: 
    
    """
    configfile=open(inputfile, 'r')
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


def write_collada_file(T,N,ind,name,r,g,b):
    """
    Exports a vertex array, a normal array, and indices array to a collada file using pycollada.
    
    Args:
      T (numpy array [:,3], float): vertices (x,y,z) of triangles.
      N (numpy array [:,3], float): vertex normals.
      ind (numpy array [:,2])     : Description of indices of T and N are collected in to triangles
      name                        : text string providing descriptive name of the collada model
      
    """
    # Create Collada Object and writer to tmp file
    mesh = Collada()
    effect = material.Effect("effect0", [], "phong", diffuse=(r,g,b), specular=(0,1,0))
    mat = material.Material("material0", "mymaterial", effect)
    mesh.effects.append(effect)
    mesh.materials.append(mat)
      
    vert_src = source.FloatSource("triverts-array", np.array(T), ('X', 'Y', 'Z'))
    normal_src = source.FloatSource("trinormals-array", np.array(N), ('X', 'Y', 'Z'))

    geom = geometry.Geometry(mesh, "geometry0", "mytri", [vert_src, normal_src])

    input_list = source.InputList()
    input_list.addInput(0, 'VERTEX', "#triverts-array")
    input_list.addInput(1, 'NORMAL', "#trinormals-array")

    triset = geom.createTriangleSet(ind, input_list, "materialref")
    geom.primitives.append(triset)
    mesh.geometries.append(geom)

    matnode = scene.MaterialNode("materialref", mat, inputs=[])
    geomnode = scene.GeometryNode(geom, [matnode])
    node = scene.Node("node0", children=[geomnode])

    myscene = scene.Scene("myscene", [node])
    mesh.scenes.append(myscene)
    mesh.scene = myscene

    mesh.write(name)


# Parse input file ##############################
(names,data)=parse_config(args.input)

Ellipsoids={}
for i in range(len(names)):
    
    # instantiate #####################################
    Ellipsoids[i]=Icosahedron(ElRes,names[i])
    
    # re-shape ########################################
    Ellipsoids[i].stretch(data[names[i]]['A'],
                          data[names[i]]['B'],
                          data[names[i]]['C'])
    
    #Define Rotations ################################
    alpha=data[names[i]]['alpha'] 
    beta =data[names[i]]['beta'] 
    gamma=data[names[i]]['gamma']

    u=np.array([1.,0.,0.])
    Ellipsoids[i].rotate_about_u(alpha,u)
    
    u=np.array([0.,math.cos(alpha),(+1)*math.sin(alpha),0.])
    Ellipsoids[i].rotate_about_u(beta,u)

    u=np.array([math.sin(beta)*math.cos(alpha),
                (-1.)*math.sin(alpha)*math.cos(beta),
                math.cos(alpha)*math.cos(beta)])            
    .rotate_about_u(gamma,u)    
   
#    Ellipsoids[i].rotate_eulerZYX(alpha,0.0,gamma)
      
    # Write .dae files ###############################
    name='./'+Ellipsoids[i].name+'.dae'
    write_collada_file(Ellipsoids[i].TP,
                       Ellipsoids[i].NP,
                       Ellipsoids[i].indices,
                       name,
                       data[names[i]]['red'],
                       data[names[i]]['green'],
                       data[names[i]]['blue'])


# Create a KML document #########################
kml = Kml()
kml.document.name = "Ellipsoids"
                   
for i in range(len(names)):
    mod = kml.newmodel(altitudemode=AltitudeMode.relativetoground,
                       location='<longitude>'+repr(data[names[i]]['lon'])+'</longitude>'+
                                '<latitude>'+repr(data[names[i]]['lat'])+'</latitude>'+
                                '<altitude>'+repr(data[names[i]]['alt'])+'</altitude>',
                       visibility=1,
                       name=names[i]
                       )
    mod.link.href=('files/'+Ellipsoids[i].name+'.dae')
    kml.addfile('./'+Ellipsoids[i].name+'.dae')

kml.savekmz(args.output)
if (nokeepfiles):
    # Remove all intermediate Collada Files
    for i in range(len(names)):
       os.remove('./'+Ellipsoids[i].name+'.dae')
