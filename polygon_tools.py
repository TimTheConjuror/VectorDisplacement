
"""
A few polygon related utilities

"""

import maya.cmds as cmds
import maya.api.OpenMaya as om


def get_mesh_polygon_iter(meshName):
    """
    Args:
        meshName (string) : name of the mesh
    Returns:
        polygon_iter (om.MItMeshPolygon) : the maya polygon iterator
    """

    dagShp = getDagShape(meshName)
    polygon_iter = om.MItMeshPolygon(dagShp)

    return polygon_iter

def get_mesh_object(meshName):
    """
    Args:
        meshName (string) : name of the mesh
    Returns:
        meshObj (om.MFnMesh) : the maya mesh object
    """

    dagShp = getDagShape(meshName)
    meshObj = om.MFnMesh(dagShp)

    return meshObj


def getDagShape(meshName):
    """
    Args:
        meshName (string) : the name of the mesh
    Returns:
        dag (OpenMaya Dag Path) : the path based identifier for maya
    """

    # presume a transform string name has been passed in.

    sel = om.MSelectionList()
    sel.add(meshName)

    dag = sel.getDagPath(0)

    dag.extendToShape()

    # test
    if dag.apiType() != 296:
        om.MGlobal.displayError("you must provide this method with a polygon mesh.")
        return None

    return dag