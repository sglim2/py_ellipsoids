Overview
========
py_ellipsoids takes a file defining one or more ellipsoid data, creates a 
Collada object (using pyCollada), and exports a KMZ file (using the 
SimpleKML python package) that includes that Collada object.



Installation
============
Some python python packages are necessary for successful execution. The 
necessary packages can be found in the requirements.txt file. To install:
```
pip install -r requirements.txt
```


Compatibility
=============
py_ellipsoids has been tested using python 2.7.8+ and python 3.4.1+.



Usage
=====
Instructions for execution can be obtained by running the command:
```
python src/py_ellipsoids.py -h 
```


The Config file
===============
The config file is a plain text file consisting of a table of values, with 
headings. Each row defines an ellipsoid, and each column describes properties 
of that ellipsoid. 
The first column has no heading, and is the ellipsoid name or description (no 
spaces permitted). The remaining columns each have a heading, and may be in 
any order - descriptions are given:

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

