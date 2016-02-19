#!/bin/python

import sys
import math
import numpy as np
import pandas as pd
from collada import *
from simplekml import Kml, Model, AltitudeMode, Orientation, Scale
from icosahedron import Icosahedron
import argparse, os, csv


def colours(colour):
    """
    Defines the rgb vaules of named colours
    """
    colours = {'white'  : np.array([1.00,1.00,1.00]),
               'silver' : np.array([0.75,0.75,0.75]),
               'grey'   : np.array([0.50,0.50,0.50]),
               'gray'   : np.array([0.50,0.50,0.50]),
               'black'  : np.array([0.00,0.00,0.00]),
               'red'    : np.array([1.00,0.00,0.00]),
               'maroon' : np.array([0.50,0.00,0.00]),
               'yellow' : np.array([1.00,1.00,0.00]),
               'olive'  : np.array([0.50,0.50,0.00]),
               'lime'   : np.array([0.00,1.00,0.00]),
               'green'  : np.array([0.00,0.50,0.00]),
               'aqua'   : np.array([0.00,1.00,1.00]),
               'teal'   : np.array([0.00,0.50,0.50]),
               'blue'   : np.array([0.00,0.00,1.00]),
               'navy'   : np.array([0.00,0.00,0.50]),
               'fuchsia': np.array([1.00,0.00,1.00]),
               'purple' : np.array([0.50,0.00,0.50])}
   
    return colours[colour]

def parse_csv(inputfile):
    """
    Parses the config file (csv format). 
    
    The config file is a plain text csv file consisting of a table of values, 
    with headings. Each row defines an ellipsoid, and each column describes 
    properties of that ellipsoid. 
    The columns each have a heading, and may be in any order. The 
    headings are (with descriptions):
    description (The name of the ellipsoid. When loaded into google-earth, this
                 forms the 'Name' of the object in the 'Places' panel)
    A           (First ellipsoid semi-axis)
    B           (Second ellipsoid semi-axis)
    C           (Third ellipsoid semi-axis)
    lat         (Latitude position of the centre of the ellipsoid)
    lon         (Longitude position of the centre of the ellipsoid)
    alt         (Altitude position of the centre of the ellispsoid, relative to
                 the ground)
    alpha       (Pitch of the ellipsoid, in degrees, about the y-axis (S-N))
    beta        (Yaw of ellipsoid, in degrees, about the z-axis (up-altitude))
    gamma       (Roll of the ellipsoid, in degrees, about the ellipsoids
                 long semi-axis)
    colour      (colour assigned to the ellipsoid. Any one of the pre-defined
                 colours is allowed)
        
    Rotations of the ellipsoid are performed in the order: rotation about  
    y-axis, then z, then the line of the ellipsoid's long semi-axis.
    
        
    Returns:
      data:
      name: 
    
    """
    df = pd.read_csv(inputfile)
    return df


def write_collada_file(T,N,ind,name,r,g,b,t):
    """
    Exports a vertex array, a normal array, and indices array to a collada file using pycollada.
    
    Args:
      T (numpy array [:,3], float): vertices (x,y,z) of triangles.
      N (numpy array [:,3], float): vertex normals.
      ind (numpy array [:,2])     : Description of indices of T and N are collected in to triangles
      name                        : text string providing descriptive name of the collada model
      r,g,b                       : colour components (r,g,b) of the ellipsoid (range 0..1)
      
    """
    # Create Collada Object and writer to tmp file
    mesh = Collada()
    effect = material.Effect("effect0", [], "phong", diffuse=(r,g,b), transparent=(r,g,b), transparency=t)
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

def main(input, output, ElRes, nokeepfiles=True):
    

    # Parse input file ##############################
    data=parse_csv(input)

    Ellipsoids={}
    for i in range(len(data)):
    
        # instantiate #####################################
        Ellipsoids[i]=Icosahedron(ElRes,data['description'][i])
    
        # re-shape ########################################
        ax=([data['A'][i],data['B'][i],data['C'][i]])
        ax.sort(key=float,reverse=True)
        Ellipsoids[i].stretch(ax[0],ax[1],ax[2])
    
        #Define Rotations ################################
        #alpah is the plunge of the ellipsoid long axis
        alpha=data['alpha'][i]*math.pi/180.
        #beta and gamma are modified to make them measured relative to North at 0 with clockwise positive
        #beta is the plunge of the ellipsoid long axis
        beta =math.pi/2.-data['beta'][i] *math.pi/180.
        gamma1=math.pi/2.-data['gamma'][i]*math.pi/180.
        #gamma is derived from gamma1 using the strike and dip of the ellipsoid AB plane
        #gamma1 is the strike of the AB plane
        gamma=-1*math.atan(-1*math.tan(math.pi/4.)*math.cos(gamma1-beta))
        
        # Rotate ellipsoid to match user-defined orientation 
        Ellipsoids[i].rotate_AlphaBetaGamma(alpha,beta,gamma) 
        
        # Rotate ellipsoid to match google-earth coordinates
        Ellipsoids[i].rotate_eulerXY(math.pi,math.pi/2.)
                
        # Write .dae files ###############################
        name='./'+Ellipsoids[i].name+'.dae'
        c=colours(data['colour'][i])
        t=(1.0)*data['transparency'][i].item()
        write_collada_file(Ellipsoids[i].TP,
                           Ellipsoids[i].NP,
                           Ellipsoids[i].indices,
                           name,
                           c[0],c[1],c[2],t)


    # Create a KML document #########################
    kml = Kml()
    kml.document.name = "Ellipsoids"
                   
    for i in range(len(data)):
        mod = kml.newmodel(altitudemode=AltitudeMode.relativetoground,
                           location='<longitude>'+repr(data['lon'][i])+'</longitude>'+
                                    '<latitude>'+repr(data['lat'][i])+'</latitude>'+
                                    '<altitude>'+repr(data['alt'][i])+'</altitude>',
                           visibility=1,
                           name=data['description'][i]
                           )
        mod.link.href=('files/'+Ellipsoids[i].name+'.dae')
        kml.addfile('./'+Ellipsoids[i].name+'.dae')

    kml.savekmz(output)
    if (nokeepfiles):
        # Remove all intermediate Collada Files
        for i in range(len(data)):
            os.remove('./'+Ellipsoids[i].name+'.dae')


if __name__ == "__main__":
        
    # Parse input args ############
    parser = argparse.ArgumentParser(description='Builds user-defined ellipsoids as Collada objects, and outputs a google kmz file containing these objects.')
    #parser.add_argument('-i', '--input', nargs="?", help='The input file.')
    parser.add_argument('input', help='The input file.')
    parser.add_argument('output', help='The output file. Output is in KMZ format')
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
    
    main(args.input, args.output, ElRes, nokeepfiles)

