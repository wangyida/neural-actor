import os
import sys
import json
import numpy
import math
import bpy
import mathutils
from mathutils import Matrix, Vector
def delete_objects():
    """
    Deletes all objects in the current scene
    deleteListObjects = [
        'MESH', 'CURVE', 'SURFACE', 'META', 'FONT', 'HAIR', 'POINTCLOUD',
        'VOLUME', 'GPENCIL', 'ARMATURE', 'LATTICE', 'EMPTY', 'LIGHT',
        'LIGHT_PROBE', 'CAMERA', 'SPEAKER'
    ]
    """
    deleteListObjects = ['LIGHT', 'LIGHT_PROBE', 'CAMERA']

    # deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # select all objects in the scene to be deleted
    for o in bpy.context.scene.objects:
        for i in deleteListObjects:
            if o.type == 'MESH':
                o.select_set(False)
            else:
                o.select_set(True)

    # deletes all selected objects in the scene:
    bpy.ops.object.delete()


def update_camera(camera,
                  focus_point=mathutils.Vector((0.0, 0.0, 0.0)),
                  distance=3.0):
    """
    Focus the camera to a focus point and place the camera at a specific distance from that
    focus point. The camera stays in a direct line with the focus point.

    :param camera: the camera object
    :type camera: bpy.types.object
    :param focus_point: the point to focus on (default=``mathutils.Vector((0.0, 0.0, 0.0))``)
    :type focus_point: mathutils.Vector
    :param distance: the distance to keep to the focus point (default=``10.0``)
    :type distance: float
    """
    looking_direction = focus_point - camera.location
    rot_quat = looking_direction.to_track_quat('Z', 'Y')

    camera.rotation_euler = rot_quat.to_euler()

    # use * instead of @ for Blender < 2.8
    camera.location = rot_quat @ mathutils.Vector((0.0, 0.0, distance))
    return camera


def add_light(position=(2, 2, 2), name="LIGHT_sample"):
    # create light datablock, set attributes
    light_data = bpy.data.lights.new(name=name, type='AREA')
    light_data.energy = 400

    # create new object with our light datablock
    light_object = bpy.data.objects.new(name=name, object_data=light_data)

    # link light object
    bpy.context.collection.objects.link(light_object)

    # make it active
    bpy.context.view_layer.objects.active = light_object

    # change location
    light_object.location = position

    # update scene, if needed
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()


#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------


# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x


# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


# build intrinsic camera parameters from Blender camera data
#
# see notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width,
                                        camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit, scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px)
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0  # only use rectangular pixels

    K = Matrix(((s_u, skew, u_0), (0, s_v, v_0), (0, 0, 1)))
    return K


# returns camera rotation and translation matrices from Blender.
#
# there are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    print(location)
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv @ R_world2bcam
    T_world2cv = R_bcam2cv @ T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((R_world2cv[0][:] + (T_world2cv[0], ),
                 R_world2cv[1][:] + (T_world2cv[1], ),
                 R_world2cv[2][:] + (T_world2cv[2], )))

    # cv2world
    RT_4x4 = Matrix((R_world2cv[0][:] + (T_world2cv[0], ),
                     R_world2cv[1][:] + (T_world2cv[1], ),
                     R_world2cv[2][:] + (T_world2cv[2], ), (0, 0, 0, 1)))
    RT_cv2world = RT_4x4
    RT_cv2world.invert()

    # Get the caemra to world matrix
    RT_cv2world = Matrix(RT_cv2world[0:3])
    R_cv2world = Matrix(
        (RT_cv2world[0][0:3], RT_cv2world[1][0:3], RT_cv2world[2][0:3]))
    T_cv2world = Vector(
        (RT_cv2world[0][3], RT_cv2world[1][3], RT_cv2world[2][3]))

    return RT, Matrix((R_world2cv)), Vector(
        (T_world2cv)), RT_cv2world, R_cv2world, T_cv2world


# RQ decomposition of a numpy matrix, using only libs that already come with
# blender by default
#
# Author: Ricardo Fabbri
# Reference implementations:
#   Oxford's visual geometry group matlab toolbox
#   Scilab Image Processing toolbox
#
# Input: 3x4 numpy matrix P
# Returns: numpy matrices r,q
def rf_rq(P):
    P = P.T
    # numpy only provides qr. Scipy has rq but doesn't ship with blender
    q, r = numpy.linalg.qr(P[::-1, ::-1], 'complete')
    q = q.T
    q = q[::-1, ::-1]
    r = r.T
    r = r[::-1, ::-1]

    if (numpy.linalg.det(q) < 0):
        r[:, 0] *= -1
        q[0, :] *= -1
    return r, q


# Input: P 3x4 numpy matrix
# Output: K, R, T such that P = K*[R | T], det(R) positive and K has positive diagonal
#
# Reference implementations:
#   - Oxford's visual geometry group matlab toolbox
#   - Scilab Image Processing toolbox
def KRT_from_P(P):
    N = 3
    H = P[:, 0:N]  # if not numpy,  H = P.to_3x3()

    [K, R] = rf_rq(H)

    K /= K[-1, -1]

    # from http://ksimek.github.io/2012/08/14/decompose/
    # make the diagonal of K positive
    sg = numpy.diag(numpy.sign(numpy.diag(K)))

    K = K * sg
    R = sg * R
    # det(R) negative, just invert; the proj equation remains same:
    if (numpy.linalg.det(R) < 0):
        R = -R
    # C = -H\P[:,-1]
    C = numpy.linalg.lstsq(-H, P[:, -1])[0]
    T = -R * C
    return K, R, T


# Creates a blender camera consistent with a given 3x4 computer vision P matrix
# Run this in Object Mode
# scale: resolution scale percentage as in GUI, known a priori
# P: numpy 3x4
def get_blender_camera_from_3x4_P(P, scale):
    # get krt
    K, R_world2cv, T_world2cv = KRT_from_P(numpy.matrix(P))

    scene = bpy.context.scene
    sensor_width_in_mm = K[1, 1] * K[0, 2] / (K[0, 0] * K[1, 2])
    sensor_height_in_mm = 1  # doesn't matter
    resolution_x_in_px = K[0, 2] * 2  # principal point assumed at the center
    resolution_y_in_px = K[1, 2] * 2  # principal point assumed at the center

    s_u = resolution_x_in_px / sensor_width_in_mm
    s_v = resolution_y_in_px / sensor_height_in_mm
    # TODO include aspect ratio
    f_in_mm = K[0, 0] / s_u
    # recover original resolution
    scene.render.resolution_x = int(resolution_x_in_px / scale)
    scene.render.resolution_y = int(resolution_y_in_px / scale)
    scene.render.resolution_percentage = scale * 100

    # Use this if the projection matrix follows the convention listed in my answer to
    # https://blender.stackexchange.com/questions/38009/3x4-camera-matrix-from-blender-camera
    R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))

    # Use this if the projection matrix follows the convention from e.g. the matlab calibration toolbox:
    # R_bcam2cv = Matrix(
    #     ((-1, 0,  0),
    #      (0, 1, 0),
    #      (0, 0, 1)))

    R_cv2world = R_world2cv.T
    rotation = Matrix(R_cv2world.tolist()) * R_bcam2cv
    location = -R_cv2world * T_world2cv

    # create a new camera
    bpy.ops.object.add(type='CAMERA', location=location)
    ob = bpy.context.object
    ob.name = 'CamFrom3x4PObj'
    cam = ob.data
    cam.name = 'CamFrom3x4P'

    # Lens
    cam.type = 'PERSP'
    cam.lens = f_in_mm
    cam.lens_unit = 'MILLIMETERS'
    cam.sensor_width = sensor_width_in_mm
    ob.matrix_world = Matrix.Translation(location) * rotation.to_4x4()

    #     cam.shift_x = -0.05
    #     cam.shift_y = 0.1
    #     cam.clip_start = 10.0
    #     cam.clip_end = 250.0
    #     empty = bpy.data.objects.new('DofEmpty', None)
    #     empty.location = origin+Vector((0,10,0))
    #     cam.dof_object = empty

    # Display
    cam.show_name = True
    # Make this the current camera
    scene.camera = ob
    # bpy.context.scene.update()
    bpy.context.view_layer.update()


def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT_world2cv, R_world2cv, T_world2cv, RT_cv2world, R_cv2world, T_cv2world = get_3x4_RT_matrix_from_blender(
        cam)
    return K @ RT_world2cv, K, RT_world2cv, R_world2cv, T_world2cv
    # return K @ RT_world2cv, K, RT_cv2world, R_cv2world, T_cv2world


def setup_depth_node(home_dir, format='exr'):
    if format == 'png':
        # Configs
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        fileOutput = None
        """
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'
        bpy.context.scene.render.image_settings.use_zbuffer = True
        bpy.context.scene.render.image_settings.color_depth = '8'

        # build a range map
        map = tree.nodes.new(type="CompositorNodeMapRange")
        # from min
        map.inputs[1].default_value = 2.0
        # from max
        map.inputs[2].default_value = 3.5
        # to min
        map.inputs[3].default_value = 0
        # to max
        map.inputs[4].default_value = 1.0

        # links.new(rl.outputs[2], map.inputs[0])
        links.new(rl.outputs['Depth'], map.inputs['Value'])

        # invert
        invert = tree.nodes.new(type="CompositorNodeInvert")
        links.new(map.outputs['Value'], invert.inputs['Color'])

        # create output node
        depthViewer = tree.nodes.new('CompositorNodeViewer')
        # depthViewer.use_alpha = False

        # links.new(rl.outputs['Depth'], depthViewer.inputs[0]) # link Z to output
        links.new(invert.outputs['Color'], depthViewer.inputs['Image'])

        # use alpha from input.
        links.new(rl.outputs['Alpha'], depthViewer.inputs['Alpha'])

        # create a file output node and set the path
        fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        links.new(invert.outputs['Color'], fileOutput.inputs['Image'])
        """
    else:
        # set up rendering of depth map
        bpy.context.scene.use_nodes = True
        tree = bpy.context.scene.node_tree
        links = tree.links

        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')
        # Save real Depth in EXR files
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.use_zbuffer = True
        # depthViewer = tree.nodes.new('CompositorNodeViewer')
        # links.new(rl.outputs['Depth'], depthViewer.inputs['Image'])
        fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        links.new(rl.outputs['Depth'], fileOutput.inputs['Image'])
    return fileOutput


def write_cam_params(intri, extri, K, RT, R, T, cam_loc, cam_tag):
    # make them into lists
    K = [x for xs in K for x in xs]
    RT = [x for xs in RT for x in xs]
    R = [x for xs in R for x in xs]
    T = [x for x in T]
    cam_loc = [x for x in cam_loc]
    intri.append({
        'K_' + str(cam_tag): {
            "data": K,
            "rows": 3,
            "cols": 3,
            "dt": 'd'
        }
    })
    intri.append({
        'dist_' + str(cam_tag): {
            "data": [0.0, 0.0, 0.0, 0.0, 0.0],
            "rows": 5,
            "cols": 1,
            "dt": 'd'
        }
    })
    extri.append({
        'Rot_' + str(cam_tag): {
            "data": R,
            "rows": 3,
            "cols": 3,
            "dt": 'd'
        }
    })
    extri.append({
        'T_' + str(cam_tag): {
            "data": T,
            "rows": 3,
            "cols": 1,
            "dt": 'd'
        }
    })
    extri.append({
        'Location_' + str(cam_tag): {
            "data": cam_loc,
            "rows": 3,
            "cols": 1,
            "dt": 'd'
        }
    })
    return intri, extri
