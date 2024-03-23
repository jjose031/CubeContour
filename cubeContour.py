
from PIL import Image
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from numpy.linalg import norm
from itertools import combinations
from math import ceil
import os, sys


def listCast(l, to = 'float'):
    '''
    Casts list-like l of ints, strings, or floats to type given by \'to\' arg (default = \'float\')
    '\to\' can be \'float\', \'str\', or \'int\'
    List can have any elements be list-likes, and lists can be jagged
    Uses round() if float/(string to be converted to int) is cast to int
    If any element is not a str, float, int, or list-like, returns False;
    otherwise returns list with structure of list-like elements preserved (though all will be lists, not e.g. tuples)
    '''

    if to not in ['float', 'string', 'str', 'int']:
        print(f"\nERROR: 'to' arg must be 'float', 'str', 'string', or 'int', not '{to}'!")
        return False

    if to in ['float', 'int']:
        for x in l:
            if type(x) == str:
                try:
                    float(x)
                except ValueError:
                    print(f"\nERROR: Can't convert string '{x}' in {l} to int or float!")
                    return False

    if to == 'float':
        new = [float(x) if type(x) in [str, float, int] else listCast(x, 'float') for x in l]
    elif to in ['str', 'string']:
        new = [str(x) if type(x) in [str, float, int] else listCast(x, 'str') for x in l]
    elif to == 'int':
        new = [int(round(float(x))) if type(x) in [str, float, int] else listCast(x, 'int') for x in l]

    for x in new:
        if x == False and x != 0:
            return False
    else:
        return new

def molStringToFloat(mol):
    '''
    Returns same molecule as input but with coords as floats
    instead of strings.
    '''
    floatmol = dict()

    for m in mol:
        name, element, x, y, z = mol[m]
        floatmol[m] = [name, element, float(x), float(y), float(z)]

    return floatmol

def close(float1, float2, threshold = 1e-7):
    '''
    Return True if absolute value of difference
    between float1 and float 2 is less than
    threshold.
    '''
    if abs(float1 - float2) < threshold:
        return True
    else:
        return False

def xor(a, b):
    '''
    Returns True if a and not b
    OR b and not a;
    otherwise returns False.
    '''
    if not a and not b:
        return False
    elif a and b:
        return False
    else:
        return True

def pointCross(point1, point2, point3):
    '''
    Returns numpy array of cross product of vectors from three points (also np arrays).
    Vectors are 1->2 and 1->3, then 1->2 × 1->3.
    '''

    vector1 = point2 - point1
    vector2 = point3 - point1
    return np.cross(vector1, vector2)

def checkCollinear(vector1 = None, vector2 = None, point1 = None, point2 = None, point3 = None, threshold = 1e-9):
    '''
    Returns True if three points or two vectors are collinear, else False.
    Vectors and points are numpy arrays.
    Only takes two vectors xor three points; vectors take priority if both given.
    '''

    if type(None) not in (type(vector1), type(vector2)):
        if norm(np.cross(vector1, vector2)) == 0:
            return True
        else:
            return False

    elif type(None) not in (type(point1), type(point2), type(point3)):
        if norm(pointCross(point1, point2, point3)) <= threshold:
            return True
        else:
            return False
    else:
        print('\nERROR: Must input either 2 vectors or 3 points. Vectors take priority if given all 5.\n')

def pointsPlane(point1, point2, point3):
    '''
    Returns numpy array of the plane containing three given points (as np arrays)
    Plane is in the form of [A, B, C, D] for Ax + By + Cz + D = 0.
    Returns False if three points are collinear.
    '''

    orthov = pointCross(point1, point2, point3)
    if norm(orthov) == 0:
        print(f'\nERROR:\tThe three points are collinear!\nPoint 1: {point1}\nPoint 2: {point2}\nPoint 3: {point3}\n')
        return False

    d = -1 * np.dot(orthov, point1)
    return np.array([orthov[0], orthov[1], orthov[2], d])

def atomsToPlane(mol, atom1, atom2, atom3):
    '''
    Given molecule and 3 non-collinear atoms, returns plane containing atoms.
    '''
    a1, a2, a3 = np.array(mol[atom1][2:]), np.array(mol[atom2][2:]), np.array(mol[atom3][2:])
    return pointsPlane(a1, a2, a3)

def intersectionTwoLines(line1, line2, verbose = True):
    '''
    Returns parameters t of two lines
    to get to point of intersection.
    '''
    swap = False
    vec1, point1 = line1
    vec2, point2 = line2

    if checkCollinear(vec1, vec2):

        x1,y1,z1 = point1
        x2,y2,z2 = point2
        a,b,c = vec2
        ts, matches = [], []

        if a != 0:
            t1 = (x1-x2)/a
            ts.append(t1)
        else:
            xMatch = close(x1, x2)
            matches.append(xMatch)

        if b != 0:
            t2 = (y1-y2)/b
            ts.append(t2)
        else:
            yMatch = close(y1, y2)
            matches.append(yMatch)

        if c!= 0:
            t3 = (z1-z2)/c
            ts.append(t3)
        else:
            zMatch = close(z1, z2)
            matches.append(zMatch)

        if not all(matches): # will be true if matches is empty
            if verbose:
                print(f'\nLines\n{line1}\nand\n{line2} are parallel.\n')
            return -1

        for ti in ts:
            for tj in ts:
                if not close(ti, tj):
                    if verbose:
                        print(f'\nLines\n{line1}\nand\n{line2} are parallel.\n')
                    return -1

        if verbose:
            print(f'\nLines\n{line1}\nand\n{line2} are coincident.\n')
        return -2


    vx1,vy1,vz1 = vec1
    vx2,vy2,vz2 = vec2
    x01,y01,z01 = point1
    x02,y02,z02 = point2

    if not close(vx2, 0) or not close(vx1, 0):

        if close(vx2, 0):
            vec1, point1 = line2
            vec2, point2 = line1
            vx1,vy1,vz1 = vec1
            vx2,vy2,vz2 = vec2
            x01,y01,z01 = point1
            x02,y02,z02 = point2
            swap = True

        if not close(vy1, vy2*vx1/vx2):
            t1 = (vy2*(x01-x02)/vx2 - y01 + y02)/(vy1 - (vy2*vx1/vx2))
            t2 = (vx1*t1 + x01 - x02)/vx2

            if not close(vz1*t1 + z01, vz2*t2 + z02):
                if verbose:
                    print(f'\nLines\n{line1}\nand\n{line2} are skew.\n')
                return -1

        else:

            if not close(y01-y02, (x01-x02)*(vy2/vx2)):
                if verbose:
                    print(f'\nLines\n{line1}\nand\n{line2} are skew.\n')
                return -1

            if not close(vz1, vz2*vx1/vx2):
                t1 = (vz2*(x01-x02)/vx2 - z01 + z02)/(vz1 - (vz2*vx1/vx2))
                t2 = (vx1*t1 + x01 - x02)/vx2

                if not close(vy1*t1 + y01, vy2*t2 + y02):
                    if verbose:
                        print(f'\nLines\n{line1}\nand\n{line2} are skew.\n')
                    return -1

            else:
                if verbose:
                    print(f'\nLines\n{line1}\nand\n{line2} are parallel. SOMETHING WENT WRONG.\n')
                return -1

    else:

        if close(vz2, 0):
            vec1, point1 = line2
            vec2, point2 = line1
            vx1,vy1,vz1 = vec1
            vx2,vy2,vz2 = vec2
            x01,y01,z01 = point1
            x02,y02,z02 = point2
            swap = True

        if not close(vy1, vy2*vz1/vz2):
            t1 = (vy2*(z01-z02)/vz2 - y01 + y02)/(vy1 - (vy2*vz1/vz2))
            t2 = (vz1*t1 + z01 - z02)/vz2

            if not close(vx1*t1 + x01, vx2*t2 + x02):
                if verbose:
                    print(f'\nLines\n{line1}\nand\n{line2} are skew.\n')
                return -1

        else:
            if verbose:
                print(f'\nLines\n{line1}\nand\n{line2} are parallel. SOMETHING WENT WRONG.\n')
            return -1

    if swap:
        t1, t2 = t2, t1

    return t1,t2

def ptTwoLines(line1, line2, verbose = True):
    '''
    Return point of intersection of two lines.
    '''
    result = intersectionTwoLines(line1, line2, verbose = verbose)
    if result in [-1, -2]:
        return result
    else:
        t1, t2 = result
    vec, p0 = line1

    pt = vec*t1 + p0

    return pt

def normalizeVectors(vectorList):
    '''
    Normalize any number of vectors.
    '''
    normal = []

    for v in vectorList:
        normed = v/norm(v)
        normal.append(normed)

    return normal

def vectorOntoVector(v, u):
    '''
    Projects vector v onto vector u.
    '''
    v,u = np.array(v), np.array(u)
    scalar = np.dot(v,u)/np.dot(u,u)
    proj = scalar * u
    return proj

def gramSchmidt(vectors):
    '''
    Generates n orthonormal vectors from n non-collinear vectors.
    '''
    orthoVectors = []

    for k, v in enumerate(vectors):

        if k == 0:
            orthoVectors.append(v)
        else:
            projs = [-1*vectorOntoVector(v, orthoVectors[j-1]) for j in range(1, k+1)]

            if len(projs) > 1:
                u = np.add(v, addVectors(projs))
            else:
                u = np.add(v, projs[0])
            orthoVectors.append(u)

    normed = normalizeVectors(orthoVectors)
    return normed

def intersectionTwoPlanes(plane1, plane2):
    '''
    Return line (vector plus point)
    of intersection between plane1 and plane2.
    '''
    plane1, plane2 = np.array(plane1), np.array(plane2)
    a1,b1,c1,d1 = plane1
    a2,b2,c2,d2 = plane2
    if checkCollinear(vector1 = np.array([a1,b1,c1]), vector2 = np.array([a2,b2,c2])):
        print('Planes are parallel.')
        return -1

    if a1 != 0 or a2 != 0:

        if a2 == 0:
            plane1, plane2 = plane2, plane1
            a1,b1,c1,d1 = plane1
            a2,b2,c2,d2 = plane2

        r = a1/a2
        denom = b1 - r*b2
        denom2nd = c1-r*c2

        if denom != 0:
            z0 = 0
            vz = 1
            y0 = (r*d2-d1)/denom
            vy = (r*c2-c1)/denom

            a2,b2,c2,d2 = plane2
            x0 = -(d2/a2) - b2*y0/a2
            vx = -(c2/a2) - b2*vy/a2

        elif denom2nd != 0:
            z0 = (r*d2-d1)/denom2nd
            vz = 0
            y0 = 0
            vy = 1

            if a1 != a2:
                x0 = (d2-d1-z0*(c1-c2))/(a1-a2)
                vx = (b2-b1)/(a1-a2)

            else:
                x0 = (-d2-z0*c2)/a2
                vx = -b2/a2

        else:
            print('Something went wrong. Planes are parallel.')
            return -1

    else:

        if b1 == 0 and b2 == 0:
            print('Something went wrong. Planes are parallel.')
            return -1

        if b2 == 0:
            plane1, plane2 = plane2, plane1
            a1,b1,c1,d1 = plane1
            a2,b2,c2,d2 = plane2

        x0 = 0
        vx = 1
        vy = 0
        vz = 0

        r = b1/b2
        denom = c1 - r*c2
        if denom == 0:
            print('Something went wrong. Planes are parallel.')
            return -1

        z0 = (r*d2-d1)/denom
        y0 = (-d2 - c2*z0)/b2

    return np.array([vx, vy, vz]), np.array([x0, y0, z0])

def threePtsOnPlane(plane):
    '''
    Return three non-collinear points on a given plane.
    Plane in form [a,b,c,d]
    for ax + by + cz + d = 0.
    '''
    a,b,c,d = plane

    if c != 0:
        p1 = [0, 0, -d/c]
        p2 = [0, 1, (-d-b)/c]
        p3 = [1, 0, (-d-a)/c]

    elif b != 0:
        p1 = [0, -d/b, 0]
        p2 = [0, (-d-c)/b, 1]
        p3 = [1, (-d-a)/b, 0]

    elif a != 0:
        p1 = [-d/a, 0, 0]
        p2 = [(-d-b)/a, 1, 0]
        p3 = [(-d-c)/a, 0, 1]

    return np.array(p1), np.array(p2), np.array(p3)

def planeTwoOrthoVectors(plane, normalize=True):
    '''
    Return two normal non-collinear vectors on a given plane.
    Plane in form [a,b,c,d]
    for ax + by + cz + d = 0.
    '''
    a,b,c,d = plane
    p1, p2, p3 = threePtsOnPlane(plane)
    v1 = p2 - p1
    v2 = p3 - p1
    if normalize:
        v1,v2 = normalizeVectors([v1,v2])
    return v1, v2

def cubeLimits(file):
    '''
    Retrieve max and min values for x,y,z coords
    in cube file.
    ASSUMES BASIS VECTORS ARE ON X,Y,Z AXES
    '''
    with open(file, 'r') as f:
        lines = f.readlines()[2:6]

    x0,y0,z0 = listCast(lines[0].split()[1:4])

    xn,xx = listCast(lines[1].split()[0:2])
    yn,yy = listCast(lines[2].split()[0:3:2])
    zn,zz = listCast(lines[3].split()[0:4:3])

    xmax = x0 + xn*xx
    ymax = y0 + yn*yy
    zmax = z0 + zn*zz

    return x0, xmax, y0, ymax, z0, zmax

def cubeVectors(file):
    '''
    Retrieve three directional vectors from cube file.
    ASSUMES BASIS VECTORS ARE ON X,Y,Z AXES
    '''
    with open(file, 'r') as f:
        lines = f.readlines()[2:6]

    xn,xx = listCast(lines[1].split()[0:2])
    yn,yy = listCast(lines[2].split()[0:3:2])
    zn,zz = listCast(lines[3].split()[0:4:3])

    return xn, xx, yn, yy, zn, zz

def xyzmol(xyzfile, molName = 'molecule1'):
    '''
    Converts .xyz file to molecule dict with structure:
    {ID : [MOLECULE NAME, NUCLEUS, X, Y, Z]}.
    '''
    if not xyzfile.endswith('.xyz'):
        print(f"\nERROR: Need a .xyz file!\n")
        return False

    with open(xyzfile, 'r') as f:
        r = f.readlines()

    molecule = dict()

    for idx, line in enumerate(r[2:]):
        if line.strip() != '':
            molecule[idx] = [molName]
            for n in range(4):
                molecule[idx].append(line.split()[n])
        else:
            break
    return molecule

def planeGridGen(cubeFile, fileContainingMol, atomsDefiningPlane, Doffset = 0, verbose = True):
    '''
    Return two orthogonal vectors and a starting point, giving the plane to map.
    '''
    if verbose:
        print(f"\nBeginning planeGridGen() with Doffset = {Doffset}...")
        print(f"Reading cube limits from .cube file {cubeFile}...")

    xmin, xmax, ymin, ymax, zmin, zmax = cubeLimits(cubeFile)

    if verbose:
        print(f"Complete.")
        print(f"X min = {xmin}")
        print(f"X max = {xmax}")
        print(f"Y min = {ymin}")
        print(f"Y max = {ymax}")
        print(f"Z min = {zmin}")
        print(f"Z max = {zmax}")

    faces = {'top':    np.array([0, 0, 1, -zmax]),
             'bottom': np.array([0, 0, 1, -zmin]),
             'front':  np.array([0, 1, 0, -ymin]),
             'back':   np.array([0, 1, 0, -ymax]),
             'right':  np.array([1, 0, 0, -xmax]),
             'left':   np.array([1, 0, 0, -xmin])}

    faceLeftLines = {'top':    (np.array([0, 1, 0]), np.array([xmin, ymin, zmax])),
                     'bottom': (np.array([0, 1, 0]), np.array([xmin, ymin, zmin])),
                     'front':  (np.array([0, 0, 1]), np.array([xmin, ymin, zmin])),
                     'back':   (np.array([0, 0, 1]), np.array([xmin, ymax, zmin])),
                     'right':  (np.array([0, 0, 1]), np.array([xmax, ymin, zmin])),
                     'left':   (np.array([0, 0, 1]), np.array([xmin, ymax, zmin]))}

    faceFirstBoundary = {'top':    (1, ymax, ymin),
                         'bottom': (1, ymax, ymin),
                         'front':  (2, zmax, zmin),
                         'back':   (2, zmax, zmin),
                         'right':  (2, zmax, zmin),
                         'left':   (2, zmax, zmin)}

    faceBackLines = {'top':    (np.array([1, 0, 0]), np.array([xmin, ymax, zmax])),
                     'bottom': (np.array([1, 0, 0]), np.array([xmin, ymax, zmin])),
                     'front':  (np.array([1, 0, 0]), np.array([xmin, ymin, zmax])),
                     'back':   (np.array([1, 0, 0]), np.array([xmin, ymax, zmax])),
                     'right':  (np.array([0, 1, 0]), np.array([xmax, ymin, zmax])),
                     'left':   (np.array([0, 1, 0]), np.array([xmin, ymax, zmax]))}

    faceSecondBoundary = {'top':    (0, xmax, xmin),
                          'bottom': (0, xmax, xmin),
                          'front':  (0, xmax, xmin),
                          'back':   (0, xmax, xmin),
                          'right':  (1, ymax, ymin),
                          'left':   (1, ymax, ymin)}

    faceFrontLines = {'top':    (np.array([1, 0, 0]), np.array([xmin, ymin, zmax])),
                      'bottom': (np.array([1, 0, 0]), np.array([xmin, ymin, zmin])),
                      'front':  (np.array([1, 0, 0]), np.array([xmin, ymin, zmin])),
                      'back':   (np.array([1, 0, 0]), np.array([xmin, ymax, zmin])),
                      'right':  (np.array([0, 1, 0]), np.array([xmax, ymin, zmin])),
                      'left':   (np.array([0, 1, 0]), np.array([xmin, ymax, zmin]))}

    if verbose:
        print(f"\nCollecting molecule from {fileContainingMol} and generating plane from atoms {atomsDefiningPlane}...")

    mol = molStringToFloat(xyzmol(fileContainingMol))
    plane = atomsToPlane(mol, *atomsDefiningPlane)
    if type(plane) == bool:
        print(f"Atoms in molecule are collinear! Need 3 non-collinear atoms to form a plane.")
        return

    if verbose:
        print(f"Complete. Plane without Doffset: {plane}")

    plane[-1] += Doffset

    if verbose:
        print(f"Plane corrected for Doffset:     {plane}")

    point = 'noPoint'

    for face in faces:

        if verbose:
            print(f"\nWorking with prism face: {face}")

        planeFaceLine = intersectionTwoPlanes(faces[face], plane)

        if verbose:
            if type(planeFaceLine) != int:
                print(f"Line of intersection between {face} face and plane:\n{planeFaceLine}")
            elif planeFaceLine == -1:
                print(f"{face} face and plane are parallel: no intersection.")
            elif planeFaceLine == -2:
                print(f"{face} face and plane are coincident.")

        nextFace = False

        if type(planeFaceLine) != int:

            faceLeftLine = faceLeftLines[face]
            faceLeftAndPlanePt = ptTwoLines(planeFaceLine, faceLeftLine, verbose)

            if verbose:
                print(f"\n{face} face 'left' edge selected:\n{faceLeftLine}")

                if type(faceLeftAndPlanePt) != int:
                    print(f"Point of intersection between left edge of {face} face and {face}-plane intersection:\n{faceLeftAndPlanePt}")
                elif faceLeftAndPlanePt == -1:
                    print(f"Left edge of {face} face and {face}-plane intersection do not intersect.")
                elif faceLeftAndPlanePt == -2:
                    print(f"Left edge of {face} face and {face}-plane intersection are coincident.")

            if type(faceLeftAndPlanePt) != int:
                coord, limit1upper, limit1lower = faceFirstBoundary[face]

                if faceLeftAndPlanePt[coord] - limit1upper > 1e-9:
                    exceed_limit1 = True

                    if verbose:
                        coordLetter = {0:'x',1:'y',2:'z'}[coord]
                        print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceLeftAndPlanePt[coord]}) "\
                              f"above limit {limit1upper}. Trying a different edge...")

                elif limit1lower - faceLeftAndPlanePt[coord] > 1e-9:
                    exceed_limit1 = False
                    below_limit1 = True

                    if verbose:
                        coordLetter = {0:'x',1:'y',2:'z'}[coord]
                        print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceLeftAndPlanePt[coord]}) "\
                              f"below limit {limit1upper}. Trying a different edge...")

                else:
                    exceed_limit1 = False
                    below_limit1 = False

                    if verbose:
                        coordLetter = {0:'x',1:'y',2:'z'}[coord]
                        print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceLeftAndPlanePt[coord]}) "\
                              f"within limits {limit1lower}-{limit1upper}")
                        print(f"Point {faceLeftAndPlanePt} selected.")
            else:
                exceed_limit1 = True # lines are collinear or coincident

            if exceed_limit1:

                faceBackLine = faceBackLines[face]
                faceBackAndPlanePt = ptTwoLines(planeFaceLine, faceBackLine, verbose)

                if verbose:
                    print(f"\n{face} face 'back' edge selected:\n{faceBackLine}")

                    if type(faceBackAndPlanePt) != int:
                        print(f"Point of intersection between back edge of {face} face and {face}-plane intersection:\n{faceBackAndPlanePt}")
                    elif faceBackAndPlanePt == -1:
                        print(f"Back edge of {face} face and {face}-plane intersection do not intersect.")
                        nextFace = True
                    elif faceBackAndPlanePt == -2:
                        print(f"Back edge of {face} face and {face}-plane intersection are coincident.")

                if type(faceBackAndPlanePt) != int:

                    coord2, limit2upper, limit2lower = faceSecondBoundary[face]

                    if faceBackAndPlanePt[coord2] - limit2upper > 1e-9 or limit2lower - faceBackAndPlanePt[coord2] > 1e-9:
                            nextFace = True

                            if verbose:
                                coordLetter = {0:'x',1:'y',2:'z'}[coord2]
                                if faceBackAndPlanePt[coord2] - limit2upper > 1e-9:
                                    print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceBackAndPlanePt[coord2]}) "\
                                          f"above limit {limit2upper}")
                                else:
                                    print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceBackAndPlanePt[coord2]}) "\
                                          f"below limit {limit2lower}")
                                print(f"Plane does not intersect {face} face.")

                    else:
                        point = faceBackAndPlanePt

                        if verbose:
                            coordLetter = {0:'x',1:'y',2:'z'}[coord2]
                            print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceBackAndPlanePt[coord2]}) "\
                                  f"within limits {limit2lower}-{limit2upper}")
                            print(f"Point {faceBackAndPlanePt} selected.")

            elif below_limit1:

                faceFrontLine = faceFrontLines[face]
                faceFrontAndPlanePt = ptTwoLines(planeFaceLine, faceFrontLine, verbose)

                if verbose:
                    print(f"\n{face} face 'front' edge selected:\n{faceFrontLine}")

                    if type(faceFrontAndPlanePt) != int:
                        print(f"Point of intersection between front edge of {face} face and {face}-plane intersection:\n{faceFrontAndPlanePt}")
                    elif faceFrontAndPlanePt == -1:
                        print(f"Front edge of {face} face and {face}-plane intersection do not intersect.")
                        nextFace = True
                    elif faceFrontAndPlanePt == -2:
                        print(f"Front edge of {face} face and {face}-plane intersection are coincident.")

                if type(faceFrontAndPlanePt) != int:

                    coord2, limit2upper, limit2lower = faceSecondBoundary[face]

                    if faceFrontAndPlanePt[coord2] - limit2upper > 1e-9 or limit2lower - faceFrontAndPlanePt[coord2] > 1e-9:
                            nextFace = True

                            if verbose:
                                coordLetter = {0:'x',1:'y',2:'z'}[coord2]
                                if faceFrontAndPlanePt[coord2] - limit2upper > 1e-9:
                                    print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceFrontAndPlanePt[coord2]}) "\
                                          f"above limit {limit2upper}")
                                else:
                                    print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceFrontAndPlanePt[coord2]}) "\
                                          f"below limit {limit2lower}")
                                print(f"Plane does not intersect {face} face.")
                    else:
                        point = faceFrontAndPlanePt

                        if verbose:
                            coordLetter = {0:'x',1:'y',2:'z'}[coord2]
                            print(f"\nPoint of intersection has {coordLetter}-coordinate ({faceFrontAndPlanePt[coord2]}) "\
                                  f"within limits {limit2lower}-{limit2upper}")
                            print(f"Point {faceFrontAndPlanePt} selected.")

            else:
                point = faceLeftAndPlanePt

        else:
            nextFace = True

        if not nextFace:
            break

    if type(point) == str:
        if verbose:
            print(f"\nPlane does not intersect any prism face.")
        return False

    if verbose:
        print(f"\nGathering all lines of intersection between plane and prism faces...")
    allEdges = []
    for face in faces:
        line = intersectionTwoPlanes(faces[face], plane)
        if type(line) != int:
            allEdges.append(line)

    if verbose:
        print(f"Complete. Number found: {len(allEdges)}")
        print(f"\nGenerating orthonormal basis vectors for plane...")

    # get first vector directly from line of intersection b/w
    # plane and face
    v1 = planeFaceLine[0]

    # generate two vectors on plane orthogonal to each other
    ortho_vectors = planeTwoOrthoVectors(plane)

    # if the first of these two vectors is not collinear with v1,
    # then take it; if it is, then certainly the other isn't, so take it
    if checkCollinear(v1, ortho_vectors[0]):
        v2 = ortho_vectors[1]
    else:
        v2 = ortho_vectors[0]

    # use Gram-Schmidt to make the two vectors orthogonal (and normal)
    v1, v2 = gramSchmidt([v1, v2])

    if verbose:
        print("Complete.")

    # now we have two orthogonal vectors and a point
    # to describe the plane as a grid
    return v1, v2, point, allEdges, plane

def checkPoint(point, xmin, xmax, ymin, ymax, zmin, zmax):
    '''
    Checks whether point is within
    rectangular prism defined in cubeFile.
    '''

    x,y,z = point
    if x > xmax or x < xmin:
        return False
    elif y > ymax or y < ymin:
        return False
    elif z > zmax or z < zmin:
        return False
    else:
        return True

def chooseGridDirection(cubeFile, v1, v2, point):
    '''
    Given grid vectors and starting point, determine which directions to traverse
    across the grid based on the limits of the rectangular prism of values.
    '''
    xmin, xmax, ymin, ymax, zmin, zmax = cubeLimits(cubeFile)
    xn,xx,yn,yy,zn,zz = cubeVectors(cubeFile)

    swap = []
    for v in (v1, v2):

        step = min(xx,yy,zz)/5
        stepOne, stepMinusOne = False, False
        count = 0
        while not xor(stepOne, stepMinusOne):

            stepOne = v*step + point

            if checkPoint(stepOne, xmin, xmax, ymin, ymax, zmin, zmax):
                stepOne = True
            else:
                stepOne = False

            stepMinusOne = -v*step + point

            if checkPoint(stepMinusOne, xmin, xmax, ymin, ymax, zmin, zmax):
                stepMinusOne = True
            else:
                stepMinusOne = False

            if stepOne and stepMinusOne:
                step *= 2
            elif not stepOne and not stepMinusOne:
                step /= 2

        if stepMinusOne:
            swap.append(True)
        else:
            swap.append(False)

    if swap[0]:
        v1 = -v1
    if swap[1]:
        v2 = -v2

    return v1, v2

def flattenedCube(cubeFile):
    '''
    Return 1D array of all voxel values in cube file.
    '''
    with open(cubeFile, 'r') as f:
        lines = f.readlines()

    startFrom = int(lines[2].split()[0]) + 6

    lines = lines[startFrom:]
    if lines[-1] == '\n':
        del lines[-1]

    flat = []
    for line in lines:
        sixFloats = line.split()
        for flo in sixFloats:
            flat.append(float(flo))

    return flat

def cartesianToVoxelValue(flat, point, cubevectors, cubelimits):
    '''
    Finds the value in a cube file at a point in 3D
    Cartesian coordinates.
    '''
    xmin, xmax, ymin, ymax, zmin, zmax = cubelimits
    xn, xx, yn, yy, zn, zz = cubevectors
    xVoxLength = yn * zn
    yVoxLength = zn
    x, y, z = point

    Nx = (x - xmin)//xx
    Ny = (y - ymin)//yy
    Nz = (z - zmin)//zz

    vox = int(Nx*xVoxLength + Ny*yVoxLength + Nz)

    return flat[vox]

def mriPlane(cubeFile, cubeValues, v1, v2, point, stepSize, NstepsDir1, NstepsDir2, verbose=True):
    '''
    Return points and values of plane surface to map.
    '''
    cubelimits = cubeLimits(cubeFile)
    xmin, xmax, ymin, ymax, zmin, zmax = cubelimits
    cubevectors = cubeVectors(cubeFile)
    xn, xx, yn, yy, zn, zz = cubevectors

    if verbose:
        print(f"\nBeginning mriPlane()...")

    if verbose:
        print(f"\nReading voxel values from .cube file...")

    flatValues = cubeValues

    gridx = np.arange(0, NstepsDir1+1, 1)
    gridy = np.arange(0, NstepsDir2+1, 1)

    if verbose:
        print(f"Complete. {len(flatValues)} values found.")
        print(f"\nGrid in Dir. 1 is of length {len(gridx)}")
        print(f"Grid in Dir. 2 is of length {len(gridy)}")
        print(f"\nBeginning grid scan...")

    gridv = []
    pts = []
    step1 = 0

    while step1 <= NstepsDir1:
        step2 = 0
        rowv = []

        if verbose:
            if step1*stepSize in [n*stepSize for n in range(0, NstepsDir1, NstepsDir1//5)]:
                print(f"Scan {ceil(100*step1/NstepsDir1)}% complete...")

        while step2 <= NstepsDir2:
            pt = step1*stepSize*v1 + step2*stepSize*v2 + point

            if not checkPoint(pt, xmin, xmax, ymin, ymax, zmin, zmax):
                value = np.nan
            else:
                value = cartesianToVoxelValue(flatValues, pt, cubevectors, cubelimits)
                pts.append(np.array(list(pt)+[value]))
            rowv.append(value)
            step2 += 1

        gridv.append(np.array(rowv))
        step1 += 1

    if verbose:
        print(f"Grid complete.")

    return gridx, gridy, np.array(gridv), pts

def intersectionLineCombination(lines, xmin, xmax, ymin, ymax, zmin, zmax, verbose = True):
    '''
    Return all points of intersection between lines within prism limits.
    '''
    combs = combinations(lines, 2)

    pts = []
    for comb in combs:
          intersection = ptTwoLines(*comb, verbose)
          if type(intersection) != int:
              pts.append(intersection)

    ptsInCube = []
    for pt in pts:
        if checkPoint(pt, xmin-0.01, xmax+0.01, ymin-0.01, ymax+0.01, zmin-0.01, zmax+0.01):
            ptsInCube.append(pt)

    return ptsInCube

def cartesianToPlaneBasis(pt, v1, v2, originPt, step, verbose = True):
    '''
    Express cartesian coords in scalar mulitpliers of plane basis vectors.
    '''
    line1 = (v1, originPt)
    line2 = (v2, pt)
    inter = ptTwoLines(line1, line2, verbose)
    inter1 = np.add(inter, -originPt)
    inter2 = np.add(pt, -inter)
    length1 = norm(inter1)/step
    length2 = norm(inter2)/step
    if np.dot(inter1, v1) < 0:
        length1 *= -1
    if np.dot(inter2, v2) < 0:
        length2 *= -1

    return length1, length2

def findPlaneLimits(v1, v2, originPt, step, lines, xmin, xmax, ymin, ymax, zmin, zmax, verbose=True):
    '''
    Find min and max points of grid in both basis vector directions.
    '''
    vertices = intersectionLineCombination(lines, xmin, xmax, ymin, ymax, zmin, zmax, verbose)
    if verbose:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(vertices)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    convertedVertices = []
    for vertex in vertices:
        convertedVertices.append(cartesianToPlaneBasis(vertex, v1, v2, originPt, step, verbose))

    ones = [x[0] for x in convertedVertices]
    min1, max1 = min(ones), max(ones)

    twos = [x[1] for x in convertedVertices]
    min2, max2 = min(twos), max(twos)

    return min1, max1, min2, max2

def gifFromPngs(pngList, name, duration=100):
    '''
    Make gif from pngs, then delete pngs.
    '''
    frames = []
    for i in pngList:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(f'{name}.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration, loop=0)

    for i in pngList:
      os.remove(i)

def plotlyfix(plane, cubeFile, cmap, res=100):
    '''
    Generate values for plotly.
    '''
    cubelimits = cubeLimits(cubeFile)
    xmin, xmax, ymin, ymax, zmin, zmax = cubelimits
    cubevectors = cubeVectors(cubeFile)
    flatValues = flattenedCube(cubeFile)

    a,b,c,d = plane
    xstep = (xmax-xmin)/res
    ystep = (ymax-ymin)/res

    X = np.arange(xmin, xmax, xstep)
    XI = np.arange(xmin, xmax, xstep)
    Y = np.arange(ymin, ymax, ystep)
    YI = np.arange(ymin, ymax, ystep)

    xlen = len(X)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = (-d-a*X-b*Y)/c

    if type(cmap) == str:
        cmap = plt.get_camp(cmap)
    colors = np.empty((xlen, ylen))

    for y in range(ylen):
        for x in range(xlen):
            z = Z[y, x]
            xi = XI[x]
            yi = YI[y]

            if checkPoint((xi, yi, z), xmin,xmax,ymin,ymax,zmin,zmax):
                value = cartesianToVoxelValue(flatValues, (xi,yi,z), cubevectors, cubelimits)
                v = list(cmap(value))
                v[-1] = 0.3 # alpha
                colors[y, x] = value
            else:
                colors[y, x] = 0

    return X,Y,Z,colors

def fix(plane, figure, cubeFile, molFile, cmap, res=100):
    '''
    Plot surface with matplotlib.
    '''
    ax = figure.add_subplot(1, 3, 1, projection='3d')

    dc = dict(H='grey', O='red', C='black', P='orange', N='blue')

    mol = xyzmol(molFile)
    mol = [mol[x][1:] for x in mol]
    mol2 = [x[1:] for x in mol]
    mol2 = listCast(mol2)

    for idx,x in enumerate(mol2):
        x.append(dc[mol[idx][0]])

    xatom = [i[0]*1.8897259886 for i in mol2]
    yatom = [i[1]*1.8897259886 for i in mol2]
    zatom = [i[2]*1.8897259886 for i in mol2]
    catom = [i[3] for i in mol2]
    xamin, xamax = min(xatom), max(xatom)
    yamin, yamax = min(yatom), max(yatom)
    zamin, zamax = min(zatom), max(zatom)

    cubelimits = cubeLimits(cubeFile)
    xmin, xmax, ymin, ymax, zmin, zmax = cubelimits
    cubevectors = cubeVectors(cubeFile)
    flatValues = flattenedCube(cubeFile)

    a,b,c,d = plane
    xstep = (xmax-xmin)/res
    ystep = (ymax-ymin)/res

    X = np.arange(xmin, xmax, xstep)
    XI = np.arange(xmin, xmax, xstep)
    Y = np.arange(ymin, ymax, ystep)
    YI = np.arange(ymin, ymax, ystep)

    xlen = len(X)
    ylen = len(Y)
    X, Y = np.meshgrid(X, Y)
    Z = (-d-a*X-b*Y)/c

    if type(cmap) == str:
        cmap = plt.get_camp(cmap)
    colors = np.empty((xlen, ylen, 4))

    for y in range(ylen):
        for x in range(xlen):
            z = Z[y, x]
            xi = XI[x]
            yi = YI[y]

            if checkPoint((xi, yi, z), xmin,xmax,ymin,ymax,zmin,zmax):
                value = cartesianToVoxelValue(flatValues, (xi,yi,z), cubevectors, cubelimits)
                v = list(cmap(value))
                v[-1] = 0.3 # alpha
                colors[y, x] = tuple(v)
            else:
                colors[y, x] = (0,0,0,0)

    ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)
    ax.scatter(xs=xatom, ys=yatom, zs=zatom, c=catom, s=40, alpha=0.8)
    ax.set_xlim((xamin-0.5,xamax+0.5))
    ax.set_ylim((yamin-0.5,yamax+0.5))
    ax.set_zlim((zamin-0.5,zamax+0.5))

def cubeContour(cubeFile, xyzfile, atomsDefiningPlane, plotlySlider=None, gifName=None, dStep = 1, stepSmaller = 3, verbose = False, cmap=None, duration=100):
    '''
    Main function.
    '''

    print("\nBeginning cubeContour()...")

    if cmap == None:
        nodes = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20, 1.0]
        colours = ['black','blue','purple','red','orange','yellow','white']
        cmap = LinearSegmentedColormap.from_list('cubeContour', list(zip(nodes,colours)))

    elif type(cmap) == list or type(cmap) == tuple:
        nodes, colours = cmap
        if gifName:
            cmap = LinearSegmentedColormap.from_list('cubeContour', list(zip(nodes,colours)))

    pngList, revPngList = [], []

    mins, maxs = [], []
    planeGridGenResults = dict()
    newPoints = dict()
    if verbose:
        print(f".cube file: {cubeFile}")
        print(f"Molecule file: {xyzfile}")
        print(f"Atoms to define plane: {atomsDefiningPlane}")

    safety = 0
    dValue = 0
    dVals = []
    dInc = dStep
    print("Constructing planes...")
    while safety < 1000:
        safety += 1

        if verbose:
            print(f"\nProceeding for Dcorr = {dValue}...")
            print(f"Generating basis, origin, and plane edges for this plane...")

        planeGenResult = planeGridGen(cubeFile, xyzfile, atomsDefiningPlane, dValue, verbose=verbose)

        if planeGenResult == False:
            if verbose:
                print("\nplaneGridGen() complete.")
                print(f"Plane with Dcorr value = {dValue} does not intersect prism.")

            if dInc > 0:
                dmax = dValue
                dInc = -dStep
                dValue = -dStep
                continue
            else:
                dmin = dValue
                break
        else:
            planeGridGenResults[dValue] = planeGenResult
            dVals.append(dValue)




        if dValue == 0:
            v1, v2, point, lines, plane = planeGenResult
            aa, bb, cc = plane[:3]
            newPoint = point
            newPoints[dValue] = newPoint
            if verbose:
                print(f"\nplaneGridGen() complete.")
                print(f"Basis vectors:\nv1 = {v1}\nv2 = {v2}")
                print(f"Origin = {point}")
                print(f"Plane-within-prism edges:")
                for line in lines:
                    print(line)
                print(f"Reading cube limits and vectors from .cube file {cubeFile}...")


        else:
            _, _, _, lines, plane = planeGenResult
            a,b,c,d = plane
            normalVector = np.array([aa,bb,cc])
            x0, y0, z0 = point
            t = (-d - a*x0 - b*y0 - c*z0)/(a*aa + b*bb + c*cc)
            newPoint = normalVector*t + point
            newPoints[dValue] = newPoint

            if verbose:
                print(f"\nplaneGridGen() complete.")
                print(f"Origin = {newPoint}")
                print(f"Plane-within-prism edges:")
                for line in lines:
                    print(line)
                print(f"Reading cube limits and vectors from .cube file {cubeFile}...")


        xmin, xmax, ymin, ymax, zmin, zmax = cubeLimits(cubeFile)
        _, xx, _, yy, _, zz = cubeVectors(cubeFile)
        stepSize = min(xx,yy,zz)/stepSmaller

        if verbose:
            print(f"Complete.")
            print(f"X min = {xmin}")
            print(f"X max = {xmax}")
            print(f"Y min = {ymin}")
            print(f"Y max = {ymax}")
            print(f"Z min = {zmin}")
            print(f"Z max = {zmax}")
            print(f"X component = {xx}")
            print(f"Y component = {yy}")
            print(f"Z component = {zz}")
            print(f"Using step size {stepSmaller} times smaller than smallest .cube file basis component.")
            print(f"Step size = {stepSize}")
            print(f"\nFinding plane limits...")

        min1, max1, min2, max2 = findPlaneLimits(v1, v2, newPoint, stepSize, lines,
                                                 xmin, xmax, ymin, ymax, zmin, zmax, verbose)
        mins.append((min1, min2, dValue))
        maxs.append((max1, max2, dValue))

        if verbose:
            print(f"Complete.")
            print(f"Dir. 1 min = {min1}")
            print(f"Dir. 1 max = {max1}")
            print(f"Dir. 2 min = {min2}")
            print(f"Dir. 2 max = {max2}")

        dValue += dInc


    globalMin1 = min([x[0] for x in mins])
    globalMin2 = min([x[1] for x in mins])
    globalMax1 = max([x[0] for x in maxs])
    globalMax2 = max([x[1] for x in maxs])

    NstepsDir1 = int(globalMax1 - globalMin1)
    NstepsDir2 = int(globalMax2 - globalMin2)

    if verbose:
        print(f"\nFound superlative minima and maxima.")
        print(f"\nDir. 1 min = {globalMin1}")
        print(f"Dir. 1 max = {globalMax1}")
        print(f"Dir. 2 min = {globalMin2}")
        print(f"Dir. 2 max = {globalMax2}")
        print(f"Steps needed in Dir. 1 = {NstepsDir1}")
        print(f"Steps needed in Dir. 2 = {NstepsDir2}")
        print(f"Dir. 1 = {v1}")
        print(f"Dir. 2 = {v2}")

    flatValues = flattenedCube(cubeFile)
    colormin, colormax = min(flatValues), max(flatValues)

    if plotlySlider:
        figPl = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "xy"}]])
        zs = []

    dValue = 0
    dInc = dStep
    if gifName is None:
        print("Obtaining values from .cube file...")
    else:
        print("Obtaining values from .cube file and creating pngs to form gif...")
    while dValue > dmin:

        if dValue == dmax:
            dInc = -dStep
            dValue = -dStep

            if plotlySlider:
                nNonNegative = len(zs)

        origin = newPoints[dValue] + v1*stepSize*globalMin1 + v2*stepSize*globalMin2

        if verbose:
            print(f"\nProceeding for Dcorr = {dValue}...")
            print(f"Origin = {origin}")

        results = mriPlane(cubeFile, flatValues, v1, v2, origin, stepSize, NstepsDir1, NstepsDir2, verbose)

        if gifName:

            if verbose:
                print(f"\nmriPlane() complete.")
                print(f"Plane is of shape {results[2].shape}.")
                print(f"\nPlotting data and saving as .png...")

            figMa = plt.figure(figsize=(12,6))
            ax = figMa.add_subplot(1, 3, (2,3))
            caxis = ax.pcolormesh(results[1], results[0], results[2], shading='auto', cmap=cmap, vmin=colormin, vmax=colormax)
            figMa.colorbar(caxis)
            ax.set_yticks([])
            ax.set_xticks([])

            fix(planeGridGenResults[dValue][-1], figMa, cubeFile, xyzfile, cmap)

            if dInc >= 0:
                dStrip = str(dValue).replace('.','')
                png = os.path.normpath(f"{gifName.replace('.gif','')}{dStrip}.png")
                pngList.append(png)
            else:
                dStrip = str(abs(dValue)).replace('.','')
                png = os.path.normpath(f"{gifName.replace('.gif','')}Z{dStrip}.png")
                revPngList.append(png)

            plt.tight_layout()
            plt.savefig(png)

            if verbose:
                print(f"Complete. Image saved as {png}")
                print(f"\nProcess complete for Dcorr = {dValue}.")

        if plotlySlider:

            if verbose:
                print(f"\nmriPlane() complete.")
                print(f"Plane is of shape {results[2].shape}.")
                print(f"\nPlotting data...")

            zs.append(results[2])

        dValue += dInc
        del results
        if gifName:
            del figMa, ax

    if gifName:
        gifFull = os.path.normpath(gifName)
        pngFullList = revPngList[::-1] + pngList

        if verbose:
            print(f"\nCreating gif from {len(pngFullList)} .png files...")
        else:
            print("Creating gif...")

        gifFromPngs(pngFullList, gifFull, duration)

        if verbose:
            print(f"Complete. File saved as {gifFull + '.gif'}.")
            print(f"\ncubeContour() complete.\n")

    if plotlySlider:
        print("Creating Plotly graph...")

        dc = dict(H='grey', O='red', C='black', P='orange', N='blue')
        mol = xyzmol(xyzfile)
        mol = [mol[x][1:] for x in mol]
        mol2 = [x[1:] for x in mol]
        mol2 = listCast(mol2)

        for idx,x in enumerate(mol2):
            x.append(dc[mol[idx][0]])

        xatom = [i[0]*1.8897259886 for i in mol2]
        yatom = [i[1]*1.8897259886 for i in mol2]
        zatom = [i[2]*1.8897259886 for i in mol2]
        catom = [i[3] for i in mol2]
        xamin, xamax = min(xatom), max(xatom)
        yamin, yamax = min(yatom), max(yatom)
        zamin, zamax = min(zatom), max(zatom)


        steps = []

        stepOrder = list(range(nNonNegative - len(zs), 0))[::-1] + list(range(nNonNegative))

        scale = []
        for node, colour in zip(nodes, colours):
            scale.append([node, colour])

        for i in stepOrder:
            zi = zs[i]
            figPl.add_trace(
                          go.Heatmap(
                          z=zi,
                          visible=False,
                          colorscale=scale,
                          name='colourmap'
                          ),
                          row=1,col=2
            )

            X, Y, Z, C = plotlyfix(planeGridGenResults[dVals[i]][-1], cubeFile, cmap)

            figPl.add_trace(
                            go.Surface(
                            x=X,
                            y=Y,
                            z=Z,
                            visible=False,
                            surfacecolor=C,
                            name='plane',
                            colorscale=scale,
                            opacity=0.5
                            ),
                            row=1,col=1
            )

        figPl.data[0].visible = True
        figPl.data[1].visible = True

        for i in range(len(figPl.data)):
            if i % 2 == 0:
                figPl.data[i].update(zmin=colormin, zmax=colormax)
                figPl.data[i+1].update(cmin=colormin, cmax=colormax)

        figPl.add_trace(
                      go.Scatter3d(
                      x=xatom,
                      y=yatom,
                      z=zatom,
                      visible=False,
                      mode='markers',
                      marker=dict(color=catom),
                      name='3dplot'
                      ),
                      row=1,col=1,
        )

        for i in range(len(figPl.data)-1):
            if i % 2 == 0:
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(figPl.data)},
                          {"title": "Plane position: " + str(i)}]
                )
                step["args"][0]["visible"][i] = True
                step["args"][0]["visible"][i+1] = True
                step["args"][0]["visible"][-1] = True
                steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Plane position: "},
            pad={"t": 50},
            steps=steps
        )]

        #
        figPl.for_each_trace(
            lambda trace: trace.update(colorbar_exponentformat="B") if trace.name == "colourmap" else (),
        )

        cubemin = min([xamin,yamin,zamin])
        cubemax = max([xamax,yamax,zamax])

        figPl.update_layout(sliders=sliders,
                            scene=dict(
                                        xaxis=dict(range=[cubemin - 0.5, cubemax + 0.5]),
                                        yaxis=dict(range=[cubemin - 0.5, cubemax + 0.5]),
                                        zaxis=dict(range=[cubemin - 0.5, cubemax + 0.5]),
                                        aspectmode='cube'
                            ))



        figPl.write_html(plotlySlider)
        figPl.show()

    print("contourGraph() complete.")

def cli(flags, arguments):
    '''
    For CLI use.
    '''

    # should go to else if in interactive
    if not hasattr(sys, 'ps1'):

        import getopt
        argv = sys.argv[1:]

        try:
            opts, args = getopt.getopt(argv, "".join(flags), arguments)

        except getopt.GetoptError as e:

            print(e)
            print(f"Accepted flags: {' '.join([f'-{x}'.replace(':','') for x in flags])}")
            print(f"Accepted args: {' '.join([f'--{x}'.replace('=','') for x in arguments])}")

            return

        tags = [x for x,y in opts]

        if '-h' in tags or '--help' in tags:
            print('\nCheck the README: https://github.com/jjose031/CubeContour/blob/main/README.md\n\n'\
                  'Command-line usage:\n'\
                  '-c, --cube=         : (REQUIRED) .cube file containing voxel values. The three axes must align with the x,y,z axes of the Cartesian grid.\n'\
                  '-x, --xyz=          : (REQUIRED) .xyz file containing molecule/structure of interest.\n'\
                  '-a, --atoms=        : (REQUIRED) 3 numbers separated by commas corresponding to the three atoms used to define the visualization planes.'\
                  ' Numbering of atoms is the order they appear in the .xyz file. Atoms must not be collinear.\n'\
                  '-p, --plotly=       : (defualt None) Generate .html file with interactive Plotly graph in which a slider can be used to translate the plane of visualization in the direction normal to the plane.\n'\
                  '-g, --gif=          : (default None) .gif file to generate. Gif is composed by translating the plane of visualization in the direction normal to the plane. If None, no gif is created.\n'\
                  '-d, --dstep=        : (default 1) Step size to use when translating plane of visualization. Large step size results in fewer planes (lower resolution).\n'\
                  '-s, --stepsmaller=  : (default 3) Step size to use for grid generated along visualization plane in order to retrieve voxel values from .cube file.'\
                  ' A value of n results in a step size n times smaller than the step size of the .cube file.\n'\
                  '-v, --verbose       : (no argument) Print information as the visualization is generated.\n'\
                  '-m, --cmap=         : (default None) Matplotlib colourmap to use. If None, a built-in colourmap is used.\n'\
                  '-t, --duration=     : (default 100) Duration (ms) of each frame in gif.\n'\
                  '-h, --help=         : (no argument) Print help.')

            return

        elif '-c' not in tags and '--cube' not in tags:

            print("ERROR: .cube file REQUIRED. Use -c or --cube=")
            return

        elif '-x' not in tags and '--xyz' not in tags:

            print("ERROR: .xyz file REQUIRED. Use -x or --xyz=")
            return

        elif '-a' not in tags and '--atoms' not in tags:

            print("ERROR: Choice of 3 atoms to form visualization plane REQUIRED (numbers separated by commas). Use -a or --atoms=")
            return

        else:

            shortLongConvert = {'-c':'--cube',
                                '-x':'--xyz',
                                '-a':'--atoms',
                                '-p':'--plotly',
                                '-g':'--gif',
                                '-d':'--dstep',
                                '-s':'--stepsmaller',
                                '-v':'--verbose',
                                '-m':'--cmap',
                                '-t':'--duration'}
            cliMap = dict()
            for opt, arg in opts:

                if opt in shortLongConvert.keys():
                    cliMap[shortLongConvert[opt]] = arg.replace('=','')
                else:
                    cliMap[opt] = arg.replace('=','')


            cliMap['--atoms'] = [int(n) for n in cliMap['--atoms'].split(',')]

            if '--plotly' not in cliMap.keys():
                cliMap['--plotly'] = None

            if '--gif' not in cliMap.keys():
                cliMap['--gif'] = None

            if '--dstep' not in cliMap.keys():
                cliMap['--dstep'] = 1
            else:
                cliMap['--dstep'] = int(cliMap['--dstep'])

            if '--stepsmaller' not in cliMap.keys():
                cliMap['--stepsmaller'] = 3
            else:
                cliMap['--stepsmaller'] = float(cliMap['--stepsmaller'])

            if '--verbose' in cliMap.keys():
                cliMap['--verbose'] = True
            else:
                cliMap['--verbose'] = False

            if '--cmap' not in cliMap.keys():
                cliMap['--cmap'] = None

            if '--duration' not in cliMap.keys():
                cliMap['--duration'] = 100
            else:
                cliMap['--duration'] = int(cliMap['--duration'])

            cubeContour(cliMap['--cube'],
                        cliMap['--xyz'],
                        cliMap['--atoms'],
                        cliMap['--plotly'],
                        cliMap['--gif'],
                        cliMap['--dstep'],
                        cliMap['--stepsmaller'],
                        cliMap['--verbose'],
                        cliMap['--cmap'],
                        cliMap['--duration'])

    else:
        return

accepted_flags = ('h', 'c:', 'x:', 'a:', 'p:', 'g:', 'd:', 's:', 'v', 'm:', 't:')
accepted_args = ('help', 'cube=', 'xyz=', 'atoms=', 'plotly=', 'gif=', 'dstep=', 'stepsmaller=', 'verbose', 'cmap=', 'duration=')
cli(accepted_flags, accepted_args)
