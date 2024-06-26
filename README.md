# CubeContour
Visualize Gaussian-generated .cube files.

CLI usage:

\~~~~~~

-c, --cube=         : (REQUIRED) .cube file containing voxel values. The three axes must align with the x,y,z axes of the Cartesian grid.

-x, --xyz=          : (REQUIRED) .xyz file containing molecule/structure of interest.

-a, --atoms=        : (REQUIRED) 3 numbers separated by commas corresponding to the three atoms used to define the visualization planes. Numbering of atoms is the order they appear in the .xyz file. Atoms must not be collinear.

-p, --plotly=       : (default None) Generate .html file with interactive Plotly graph in which a slider can be used to translate the plane of visualization in the direction normal to the plane.

-g, --gif=          : (default None) .gif file to generate. Gif is composed by translating the plane of visualization in the direction normal to the plane. If None, no gif is created.

-d, --dstep=        : (default 1) Step size to use when translating plane of visualization. Large step size results in fewer planes (lower resolution).

-s, --stepsmaller=  : (default 3) Step size to use for grid generated along visualization plane in order to retrieve voxel values from .cube file. A value of n results in a step size n times smaller than the step size of the .cube file.

-v, --verbose       : (no argument) Print information as the visualization is generated.

-m, --cmap=         : (default None) Matplotlib colourmap to use. If None, a built-in colourmap is used.

-t, --duration=     : (default 100) Duration (ms) of each frame in gif.

-h, --help=         : (no argument) Print help.

\~~~~~~

Usage in REPL:

contourGraph(cubeFile, xyzFile, atomsDefiningPlane, plotlySlider=None, gifName=None, dStep=1, stepSmaller=3, verbose=True, cmap=None, duration=100)

\~~~~~~

**cubeFile**: .cube file containing voxel values. The three axes must align with the x,y,z axes of the Cartesian grid.

**xyzFile**: .xyz file containing molecule/structure of interest.

**atomsDefiningPlane**: list-like of 3 ints representing non-collinear atoms (in the order in which they are given in the .xyz file), thereby defining a 2D plane. Values will be visualized in planes parallel to this plane.

**plotlySlider**: .html file to create an interactive Plotly graph in which a slider can be used to translate the plane of visualization in the direction normal to the plane.

**gifName**: name of .gif file to create (if not None). Gif is composed by translating the plane of visualization in the direction normal to the plane.

**dStep**: step size to use when translating plane of visualization. Large step size results in fewer planes (lower resolution).

**stepSmaller**: step size to use for grid generated along visualization plane in order to retrieve voxel values from .cube file. A value of n results in a step size n times smaller than the step size of the .cube file.

**verbose**: print information as the visualization is generated.

**cmap**: colour mapping to use for values. If None, a built-in map is used.

**duration**: duration (ms) of each frame in .gif.

