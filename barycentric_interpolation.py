
"""
#####################################################################
A few methods for calculating barycentric weights of a triangle:

A great explanation for barycentric coordinates by Mahmoud Youssef on youtube:
https://www.youtube.com/watch?v=6MLBP-q8pqE

"""


import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma
import maya.cmds as cmds


def getProjectionAtoB(a, b):
    """
    Args:
        a (om.MPoint) : a maya pt object
        b (om.MPoint) : a maya pt object
    Returns:
        projectionVector (om.MPoint) : how far a projects along b
    """
    return ( (a*b) / (b*b) ) * b


def get_barycentric_wts(A, B, C, pos, rec=True):

    """
    Args:
        A (om.MPoint) : the closest point of a triangle
        B (om.MPoint) : the middle point of a triangle
        C (om.MPoint) : the furthest point of a triangle
        pos (om.MPoint) : the input position we want to get weights for
    Returns:
        AWeight (float) : a percentage value of the interpolation of wts based on input positions. 
            corresponds to the A pt on the triangle.
        BWeight (float) : a percentage value of the interpolation of wts based on input positions
            corresponds to the B pt on the triangle.
        CWeight (float) : a percentage value of the interpolation of wts based on input positions
            corresponds to the C pt on the triangle.
        pos (om.MPoint) : the closest point on the triangle plane, relative to the input pos.
    """

    # barycentric calc
    AB = B - A
    AC = C - A

    # cross product of vectors
    norm = AB ^ AC

    area = norm.length() * 0.5 # getting a recursion depth error here : even when false!

    if norm.length() < 0.001:
        area = 0.0005

    if rec:
        if norm.length() > 0.001:
            pProj = getProjectionAtoB( (pos - A), norm)
            pos = om.MPoint(om.MVector(pos) - pProj)

    Cpos = C - pos
    areaposCB = ( (B - pos) ^ Cpos ).length() * 0.5
    # U
    AWeight = areaposCB / area

    areaposCA = ( (A - pos) ^ Cpos ).length() * 0.5

    # V
    BWeight = areaposCA / area
    
    CWeight = 1 - AWeight - BWeight
    
    if rec:
        if ((CWeight < 0.0 or CWeight > 1.0) or
            (BWeight < 0.0 or BWeight > 1.0) or
            (AWeight < 0.0 or AWeight > 1.0) and rec):
            # we have projected to a point off of the triangle
            # project again!  use the vector with the greater dot product, to ensure smallest angle
            
            AP = pos - A
            Vec = AB if(AB * AP > AC * AP) else AC
            # project away from triangle toward pt in space
            cNorm = Vec ^ norm
            if cNorm.length() > 0.001:
                pos = om.MPoint(om.MVector(pos) - getProjectionAtoB(AP, cNorm) )
            # recursion (only one time, if we started off of the 3 pt triangle plane)
            return get_barycentric_wts(A, B, C, pos, False)

    return (AWeight, BWeight, CWeight), pos