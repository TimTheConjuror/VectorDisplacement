
"""
a set of functions used to calculate the difference between 2 meshes, and save that as a vector displacement

"""

import imp

import OpenEXR
import numpy as np

import maya.cmds as cmds
import maya.api.OpenMaya as om

import polygon_tools as polyTools

import barycentric_interpolation as barycentric

WORKING_CLR_SET = 'working_color'

def create_displacement_map(src_mesh, tgt_mesh, image_path, image_size, 
                            pixel_extend=1, pixel_smooth=1,
                            color_set_name=WORKING_CLR_SET, flip_green_channel=True):
    """
    Args:
        src_mesh (string) : name of the original mesh, undeformed. This is the transform name, not the mesh shape.
        tgt_mesh (string) : name of the target mesh, with shape deltas making it distinct from the source.
        image_path (string/path) : the path to the exr image file which stores the deformed vector data.
        image_size (tuple<int, int>) : the width and the height of the desired image ex : (1024, 1024)
        pixel_extend (int) : how many pixels to expand the vertex color values, to increase accuracy.
        pixel_smooth (int) : how many pixels to expand a linear smoothing of the pixel extend.
        color_set_name (string) : what to call the color set used to bake down vert deltas into colors.
        flip_green_channel (bool) : unreal's space coords need an inverted green channel.
    Returns:
        norm_mag (int) : the number of times the tool reduced the intensity of the color values, in order to encapsulate all
            of the vertex offset info into one image with a color range of 0.0 - 1.0 (with 0.5 being the default value)
            This is an int where the value indicates magnitude of 10.  so if it returns 2, the intensity should be * 20
    """

    displacement = get_displacement_data(src_mesh, tgt_mesh)

    norm_mag = get_normalization_magnitude(displacement)

    print('-------------------------------')
    print('Normalization Magnitude = '+str(norm_mag))
    print('-------------------------------')

    tgtObj = polyTools.get_mesh_object(tgt_mesh)

    clrSet = create_vertex_colors(tgtObj, color_set_name, flip_green_channel, norm_mag, displacement)

    # create uv grid values
    hGridWidth = 1.0 / image_size[0] 
    vGridHeight = 1.0 / image_size[1]

    poly_iter = polyTools.get_mesh_polygon_iter(tgt_mesh)

    # exr np color array
    clr_img_array = np.zeros((image_size[1], image_size[0] , 3), dtype=np.float32)# 64?

    # lay in the base color of the map at every pixel

    # sample color values by uv grid 
    for hIndex in range(image_size[0]):
        sample_u_pos = hIndex * hGridWidth
        for vIndex in range(image_size[1]):
            #print(vIndex)
            sample_v_pos = vIndex * vGridHeight
            result_color = sample_uv_at_grid_location(tgtObj, sample_u_pos, sample_v_pos, 
                poly_iter, clrSet, hGridWidth, pixel_extend)
            # store the color, should be in a  way that works with pillow/openexr
            clrs = result_color.getColor()
            # update the exr
            clr_img_array[image_size[1] - (vIndex + 1), hIndex, 0] = clrs[0]  # Red channel
            clr_img_array[image_size[1] - (vIndex + 1), hIndex, 1] = clrs[1]  # Green channel
            clr_img_array[image_size[1] - (vIndex + 1), hIndex, 2] = clrs[2]  # Blue channel

    clr_img_array = update_pixel_extend(clr_img_array, poly_iter, tgtObj, clrSet, image_size, pixel_extend, pixel_smooth)

    # save the exr
    save_image(clr_img_array, image_path)

    return norm_mag


def create_vertex_colors(tgtObj, color_set_name, flip_green_channel, norm_mag, displacement):
    """
    Args:
        tgtObj (OpenMaya.MFnMesh) : maya mesh object
        color_set_name (string) : chosen color set name
        flip_green_channel (bool) : unreal uses inverted color in green channelf or coords
        norm_mag (int) : a factor of 10 we use to normalize the intensity fo the color
        displacement (dict) : key = faceid <int>, value = list of colors [<OpenMaya.MColor>]
    Returns:
        clrSet (string) : result name of color set
    """

    # create color set on target

    # test if color set exists.  do not create if it does.
    clrSet = None
    clrSetList = tgtObj.getColorSetNames()
    if not clrSetList or not color_set_name in clrSetList:
        clrSet = tgtObj.createColorSet(color_set_name, False)
    else:
        clrSet = color_set_name

    flip_green = 1
    if flip_green_channel:
        flip_green = -1

    current_clr_id = 0

    # loop through the face ids
    for index in displacement:
        vertColorList = displacement[index]
        for vertIndex, vertColor in enumerate(vertColorList):
            # 3 value entry  an mpoint offset value
            clr = om.MColor()

            clrR = vertColor[0] + 0.5
            clrG = (vertColor[1]*flip_green) + 0.5
            clrB = vertColor[2] + 0.5

            if norm_mag > 0:
                clrR = (vertColor[0] / (10**norm_mag) ) + 0.5
                clrG = (vertColor[1] / ((10**norm_mag)*flip_green) ) + 0.5, 
                clrB = (vertColor[2] / (10**norm_mag) ) + 0.5

            # remove tiny values that report back a tuple
            if isinstance(clrR, tuple):
                #clrR = 0.5
                clrR = clrR[0]
            if isinstance(clrG, tuple):
                #clrG = 0.5
                clrG = clrG[0]
            if isinstance(clrB, tuple):
                #clrB = 0.5
                clrG = clrG[0]

            clr.setColor([clrR, clrG, clrB])

            # set the color of the color id (which is seprate from vert ids)
            tgtObj.setColor(current_clr_id, clr, clrSet)
            # now assign the color id to a face and vertex index
            tgtObj.assignColor(index, vertIndex, current_clr_id, clrSet)
            current_clr_id += 1

    # toggle colors on
    tgtObj.displayColors = True

    return clrSet


def save_image(clr_img_array, image_path):
    """
    Args:
        clr_img_array (numpy.array) : color data for the image
        image_path (string/path) : image file location
    """

    # save the exr
    channels = { "RGB" : clr_img_array }
    header = { "compression" : OpenEXR.ZIP_COMPRESSION,
           "type" : OpenEXR.scanlineimage }

    with OpenEXR.File(header, channels) as outfile:
        outfile.write(image_path)


def get_pixel_smooth_coords(home_pixel, image_size, pixel_extend, pixel_smooth):
    """
    Args:
        home_pixel (tuple<int, int>) : pixel coordinates int he result image
        image_size (tuple<int, int>) : the width and height of the result image
        pixel_extend (int) : how far to push vert colors and border edges for accuracy
        pixel_smooth (int) : how far to further extend while linear fading into the background color.  improves image appeal.
    Returns:
        pixel_smooth_data (dict) : key = layer <int>, value = list of coordinates [tuple<int, int>]. layer is used to calculate fade amt.
    """

    pixel_smooth_data = {}

    smooth_layer = pixel_smooth

    overdraw = get_extend_coords(home_pixel, image_size, pixel_extend)

    for index in range(pixel_smooth):
        new_pixels = get_extend_coords(home_pixel, image_size, pixel_extend+pixel_smooth-index)
        if overdraw:
            for pixel in overdraw:
                if pixel in new_pixels:
                    new_pixels.remove(pixel)
        if pixel_smooth_data.keys():
            for layer in pixel_smooth_data.keys():
                for pixel in new_pixels:
                    if pixel in pixel_smooth_data[layer]:
                        pixel_smooth_data[layer].remove(pixel)
        pixel_smooth_data[smooth_layer] = new_pixels

        smooth_layer -= 1

    # data format: dict[layer_num: 2, 1, 0]: [ list of pixels ]
    return pixel_smooth_data


def get_extend_coords(home_pixel, image_size, pixel_extend):
    """
    Args:
        home_pixel (tuple<int, int>) : pixel coordinates int he result image
        image_size (tuple<int, int>) : the width and height of the result image
        pixel_extend (int) : how far to push vert colors and border edges for accuracy
    Returns:
        new_pixels (list<tuple<int, int>>) : list of new coordinates surroundign the home pixel.
    """

    new_pixels = []

    for hIndex in range(-pixel_extend, pixel_extend+1):

        for vIndex in range(-pixel_extend, pixel_extend+1):

            if hIndex == 0 and vIndex == 0:
                # were on our starting pixel:
                continue

            # test bounds of image
            if hIndex+home_pixel[0] >= image_size[0] or hIndex+home_pixel[0] < 0:
                continue
            if vIndex+home_pixel[1] >= image_size[1] or vIndex+home_pixel[1] < 0:
                continue

            # rub off corners
            if abs(hIndex) == pixel_extend and abs(vIndex) == pixel_extend and pixel_extend > 1:
                continue

            # rub off top row
            if abs(hIndex) == pixel_extend and abs(vIndex) >= max( (pixel_extend / 2)-1, 2):
                continue
            if abs(vIndex) == pixel_extend and abs(hIndex) >=  max( (pixel_extend / 2)-1, 2):
                continue

            # dig deeper into corners for larger pixel overdraw
            if abs(hIndex) > (pixel_extend / 2) + 1 and abs(vIndex) > max(pixel_extend - 2, 2):
                continue
            if abs(vIndex) > (pixel_extend / 2) + 1 and abs(hIndex) > max(pixel_extend - 2, 2):
                continue

            new_pixels.append([hIndex+home_pixel[0], vIndex+home_pixel[1]])

    return new_pixels


def update_pixel_extend(clr_img_array, poly_iter, tgtObj, clrSet, image_size, pixel_extend, pixel_smooth=0):
    """
    Args:
        clr_img_array (numpy.array) : color data input
        poly_iter (OpenMaya.MItMeshPolygon) : maya polygon iterator
        tgtObj (OpenMaya.MFnMesh) : maya mesh object
        clrSet (string) : name of the color set we are sampling
        image_size (tuple <int, int>) : width, height
        pixel_extend (int) : number of pixels we want to overlap vert colors for accuracy
        pixel_smooth (int) : bumber of pixels we want to smooth extended pixels into image background
    Returns:
        clr_img_array (numpy.array) : the updated image colors
    """

    # sample colors directly from the verts and extend to nearby pixels, to prevent mis-sampling at runtime
    poly_iter.setIndex(0)

    while not poly_iter.isDone():
        faceID = poly_iter.index()
        vertices = poly_iter.getVertices()

        for faceVertexId, globalVertexId in enumerate(vertices):

            uvGridLoc, vertClr = sample_uv_index(tgtObj, faceID, faceVertexId, 
                poly_iter, clrSet, image_size[0], image_size[1])

            clrs = vertClr.getColor()

            # update the exr
            clr_img_array[uvGridLoc[1], uvGridLoc[0], 0] = clrs[0]  # Red channel
            clr_img_array[uvGridLoc[1], uvGridLoc[0], 1] = clrs[1]  # Green channel
            clr_img_array[uvGridLoc[1], uvGridLoc[0], 2] = clrs[2]  # Blue channel

            # do the vert position overdraw
            if pixel_extend:
                new_pixels = get_extend_coords(uvGridLoc, image_size, pixel_extend)
                for h, v in new_pixels:
                    clr_img_array[v, h, 0] = clrs[0]
                    clr_img_array[v, h, 1] = clrs[1]
                    clr_img_array[v, h, 2] = clrs[2]

            # get smoothing
            if pixel_smooth:
                smooth_data = get_pixel_smooth_coords(uvGridLoc, image_size, pixel_extend, pixel_smooth)

                for index in smooth_data:
                    # how much to blend?  lets see:  ps 2 would make 1 inner, and 2 outer.
                    blend_index_amt = 1.0 / (pixel_smooth + 1) # 2 would be 0.33
                    blend_amt = 1.0 - (index * blend_index_amt) # lyr 1 = .66, lyr 2 = .33

                    pixels = smooth_data[index]
                    for pixel in pixels:
                        current_clr_r = clr_img_array[pixel[1], pixel[0], 0]
                        current_clr_g = clr_img_array[pixel[1], pixel[0], 1]
                        current_clr_b = clr_img_array[pixel[1], pixel[0], 2]

                        # new clr
                        new_r = (current_clr_r * (1.0 - blend_amt)) + (clrs[0] * blend_amt)
                        new_g = (current_clr_g * (1.0 - blend_amt)) + (clrs[1] * blend_amt)
                        new_b = (current_clr_b * (1.0 - blend_amt)) + (clrs[2] * blend_amt)

                        # set
                        clr_img_array[pixel[1], pixel[0], 0] = new_r
                        clr_img_array[pixel[1], pixel[0], 1] = new_g
                        clr_img_array[pixel[1], pixel[0], 2] = new_b

        poly_iter.next()

    return clr_img_array


def get_uv_image_grid_pos(uVal, vVal, gridWidth, gridHeight):
    """
    Args:
        uVal (float) : horizontal uv coord
        vVal (float) : vertical uv coord
        gridWidth (int) : how many pixels wide the result image is
        gridHeight (int) : how many pixels tall the result image is
    Returns:
        uGrid (int) : horizontal pixel coordinate
        vGrid (int) : vertical pixel coordinate
    """
    
    uGrid = int(float(gridWidth) * uVal)
    vGrid = gridHeight - (int(float(gridHeight) * vVal) + 1)
    return [uGrid, vGrid]


def sample_uv_index(meshFN, faceID, vertID, poly_iter, clrSet, gridWidth, gridHeight):
    """
    Args:
        meshFN (OpenMaya.MFnMesh) : mesh maya object
        faceID (int) : the id of the related face we want to sample
        vertID (int) : the id of the face vertex we want to sample
        poly_iter (OpenMaya.MItMeshPolygon) : polygon iterator maya object
        clrSet (string) : name of the color set we want to sample colors from.
        gridWidth (float) : distance from pixel to pixel (horizontal) in the result image
        gridHeight (float) : distance from pixel to pixel (vertical) in the result image
    Returns:
        pUPos (int) : horizontal pixel image coordinate
        pVPos (int) : vertical pixel image coordinate
        vertClr (OpenMaya.MColor) : color of the sampled vertex
    """
    
    # like sample at grid, but get uv and grid pos from point
    uPos, vPos = meshFN.getPolygonUV(faceID, vertID, 'map1')

    # get pixel pos
    pUPos, pVPos = get_uv_image_grid_pos(uPos, vPos, gridWidth, gridHeight)

    poly_iter.setIndex(faceID)

    # get vert color (MColor)
    vertClr = poly_iter.getColor(vertID)

    return [ [pUPos, pVPos], vertClr ]


def sample_uv_at_grid_location(meshFN, u_pos, v_pos, poly_iter, clrSet, gridWidth, pixel_extend):
    """
    Args:
        meshFN (OpenMaya.MFnMesh) : mesh maya object
        u_pos (float) : 1st entry of uv coordinates
        v_pos (float) : 2nd entry of uv coordinates
        poly_iter (OpenMaya.MItMeshPolygon) : polygon iterator maya object
        clrSet (string) : name of the color set we want to sample colors from.
        gridWidth (float) : distance from pixel to pixel in the result image
        pixel_extend (int) : how far we allow the uv border to bleed color
    Return:
        result_color (OpenMaya.MColor) : the sampled result at this uv coordinate
    """

    miss_tolerance = gridWidth * pixel_extend

    # first get 3d position from mesh obj
    uvIntArray, uvPtArray = meshFN.getPointsAtUV(u_pos, v_pos, uvSet='map1', tolerance=miss_tolerance)

    if not uvPtArray or uvPtArray.__len__() < 1:
        # return black/grey
        bColor = om.MColor()
        bColor.setColor([0.5, 0.5, 0.5, 0.0])
        #bColor.setColor([0.0, 0.0, 0.0, 1.0])
        return bColor

    uvPt = uvPtArray[0]

    # reverse call, just to get face id
    u, v, faceID = meshFN.getUVAtPoint(uvPt, uvSet='map1')

    # go to poly iter
    poly_iter.setIndex(faceID)

    # get vert locations and ids
    vertIds = poly_iter.getVertices()
    face_pt_data = {}
    for faceVertId, vId in enumerate(vertIds):

        pt = meshFN.getPoint(vId)
        face_pt_data[vId] = {}
        face_pt_data[vId]['pt'] = pt
        face_pt_data[vId]['dist'] = pt.distanceTo(uvPt)
        face_pt_data[vId]['face_vert_id'] = faceVertId

    interpData = {}

    pt_indices = list(face_pt_data.keys())
    distances = [face_pt_data[vId]['dist'] for vId in pt_indices]
    sortedPtIds = sorted(pt_indices, key=lambda x: distances[pt_indices.index(x)])

    # trim to 3 entries.
    for interpIndex in range(0, 3):

        interpData[interpIndex] = {}
        interpData[interpIndex]['dist'] = face_pt_data[sortedPtIds[interpIndex]]['dist']
        interpData[interpIndex]['pt'] = face_pt_data[sortedPtIds[interpIndex]]['pt']
        interpData[interpIndex]['id'] = sortedPtIds[interpIndex]
        interpData[interpIndex]['face_vert_id'] = face_pt_data[sortedPtIds[interpIndex]]['face_vert_id']

    # barycentric interp 3 verts.  
    bary_result = barycentric.get_barycentric_wts(interpData[0]['pt'], interpData[1]['pt'], interpData[2]['pt'], uvPt, True)

    # sample colors of 3 verts
    interpolated_color_r = 0
    interpolated_color_g = 0
    interpolated_color_b = 0

    for index, weight in enumerate([ bary_result[0][0], bary_result[0][1], bary_result[0][2] ]):

        iD = interpData[index]['face_vert_id']

        # get vert color (MColor)
        vertClr = poly_iter.getColor(iD)

        # reduce it by weight bias
        r, g, b, a = vertClr.getColor()

        # add result to finished value
        interpolated_color_r += r*(weight)
        interpolated_color_g += g*(weight)
        interpolated_color_b += b*(weight)

    # return result color.
    return_color = om.MColor()
    return_color.setColor([interpolated_color_r, interpolated_color_g, interpolated_color_b, 1.0])

    return return_color


def get_normalization_magnitude_float(dispalcement_data):
    """
    Args:
        dispalcement_data (dict) : key = faceId<int>, value = offset list per vertex on face. [<OpenMaya.MPoint>]
    Returns:
        normalization_factor (int) : the factor we use to re-create this level of vector displacement
    """

    furthest_delta = 0.0

    for faceId in dispalcement_data:
        vertColorList = dispalcement_data[faceId]
        for vertIndex, clrVec in enumerate(vertColorList):
            for value in [clrVec[0], clrVec[1], clrVec[2]]:
                if abs(value) > furthest_delta:
                    furthest_delta = abs(value)

    # clamp within + - 0.5 # posibly imbed this value int he file name.
    normalization_factor = furthest_delta * 2.0

    return normalization_factor



def get_normalization_magnitude(dispalcement_data):
    """
    Args:
        dispalcement_data (dict) : key = faceId<int>, value = offset list per vertex on face. [<OpenMaya.MPoint>]
    Returns:
        normalization_factor (int) : the factor of 10 we use to re-create this level of vector displacement
    """

    # to keep the numbers clean, we'll udpate the normalize by factors of 10

    # since they are up and down,  they'll need to fit into .5 sized slices
    furthest_delta = 0.0

    normalization_factor = 0

    for faceId in dispalcement_data:
        vertColorList = dispalcement_data[faceId]
        for vertIndex, clrVec in enumerate(vertColorList):
            for value in [clrVec[0], clrVec[1], clrVec[2]]:
                if abs(value) > furthest_delta:
                    furthest_delta = abs(value)

    #if furthest_delta >= 0.5:

    while furthest_delta >= 0.5:
        furthest_delta *= 0.1
        normalization_factor += 1

    return normalization_factor


def get_displacement_data(src_mesh, tgt_mesh):
    """
    Args:
        src_mesh (string) : name of the transform of the source mesh.
        tgt_mesh (string) : name of the transform of the target mesh.
    Returns:
        faceVertOffsetData (dict) : key = faceId <int>, value = offset list [<OpenMaya.MPoint>]
    """

    # this goes face by face and collects vertex info, organized by face id

    #grab pts  to make distance comparisons
    srcObj = polyTools.get_mesh_object(src_mesh)
    src_pts = srcObj.getPoints()

    tgtObj = polyTools.get_mesh_object(tgt_mesh)
    tgt_pts = tgtObj.getPoints()

    # use a poly iter to scrub the face verts
    poly_iter = polyTools.get_mesh_polygon_iter(tgt_mesh)
    poly_iter.setIndex(0)

    faceVertOffsetData = {}

    while not poly_iter.isDone():
        faceID = poly_iter.index()

        # these are the global vert id's not the face/vert id's
        verts = poly_iter.getVertices()

        faceVertOffsetData[faceID] = []

        for vert in verts:
            disp = tgt_pts[vert] - src_pts[vert]
            faceVertOffsetData[faceID].append(disp) 

        poly_iter.next()

    return faceVertOffsetData





