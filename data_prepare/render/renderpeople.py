import os
import sys
import json
import numpy
import bpy
import math
import mathutils
from mathutils import Matrix, Vector

path = os.getcwd()
dir = os.path.join(path, 'render')
if not dir in sys.path:
    sys.path.append(dir)

from tools import *
from bparser import ArgumentParserForBlender

if __name__ == "__main__":
    parser = ArgumentParserForBlender()
    parser.add_argument('--with_videos',
                        action='store_true',
                        help="model to extract joints from image")
    parser.add_argument('--with_images',
                        action='store_true',
                        help="model to extract joints from image")
    parser.add_argument('--with_depth',
                        action='store_true',
                        help="model to extract joints from image")
    parser.add_argument('--with_bkg',
                        action='store_true',
                        help="render with background")
    parser.add_argument('--start', type=int, default=1, help='starting frame')
    parser.add_argument('--end', type=int, default=1, help='ending frame')
    args = parser.parse_args()

    """
    add_light(position=(2, 2, 2), name='light1')
    add_light(position=(2, 1, 2), name='light2')
    add_light(position=(-2, -2, 2), name='light3')
    add_light(position=(-2, -1, 2), name='light4')
    add_light(position=(2, -2, 2), name='light5')
    add_light(position=(2, -1, 2), name='light6')
    add_light(position=(-2, 2, 2), name='light7')
    add_light(position=(-2, 1, 2), name='light8')
    """

    # NOTE: vanilla Nerf only need a single frame
    bpy.context.scene.frame_start = args.start
    bpy.context.scene.frame_end = args.end
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
        0].default_value = (0, 0, 0, 0)

    human = bpy.context.scene.objects['rp_aliyah_4d_004_dancing']
    body_shift = 0 # 0.85
    human.location -= mathutils.Vector((0, 0, body_shift))

    home_dir = os.path.expanduser('~')
    # add background
    bg_scale = 3
    if args.with_bkg:
        ground_shift = bg_scale - body_shift
        bpy.ops.mesh.primitive_cube_add(size=2*bg_scale,
                                        enter_editmode=False,
                                        align='WORLD',
                                        location=(0, 0, ground_shift),
                                        scale=(1, 1, 1))
        cube_bg = bpy.context.collection.objects['Cube']
        mat_bg = bpy.data.materials.new(name='Material')
        mat_bg.use_nodes = True
        bsdf = mat_bg.node_tree.nodes["Principled BSDF"]
        # bsdf.inputs['Base Color'].default_value = (0,1,1,1)
        texImage = mat_bg.node_tree.nodes.new('ShaderNodeTexImage')
        texImage.image = bpy.data.images.load(
            os.path.join(
                home_dir,
                'Documents/gitfarm/research-interns-2022/data_prepare/images',
                'background.png'))
        mat_bg.node_tree.links.new(bsdf.inputs['Base Color'],
                                   texImage.outputs['Color'])

        cube_bg.data.materials.append(mat_bg)

    # add camera
    camera_data = bpy.data.cameras.new(name='Camera')
    camera_object = bpy.data.objects.new('Camera', camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    camera = bpy.data.objects['Camera']
    bpy.context.scene.camera = camera

    # optional angles to position camera
    # Larger variance
    interval_x = 20
    interval_y = 10
    angle_x = list(range(0, 359, interval_x))
    angle_y = list(range(-50, -9, interval_y))
    dist_all = list(range(2, 5, 2))
    # Smaller variance
    """
    interval_x = 2
    interval_y = 2
    angle_x = list(range(0, 20, interval_x))
    angle_y = list(range(0, 20, interval_y))
    """
    intri = []
    extri = []

    num_views = len(angle_x) * len(angle_y)
    name_indexes = list(range(1, num_views + 1, 1))
    cam_tags = [str(x) for x in name_indexes]
    intri.append({"names": cam_tags})

    save_videos = args.with_videos
    save_images = args.with_images
    save_depth = args.with_depth

    # Start to go over all camera positions from the initiated position
    pi_2 = math.pi * 2
    for dist in range(len(dist_all)):
        for camid_vert in range(len(angle_y)):
            for camid_horiz in range(-1, len(angle_x)):
                if camid_horiz != -1:
                    cam_tag = dist * len(angle_x) * len(angle_y) + camid_vert * len(angle_x) + camid_horiz + 1
                    z_factor = math.cos(angle_y[camid_vert] / 360.0 * pi_2)
                    cam_x, cam_y, cam_z = math.cos(
                        angle_x[camid_horiz] / 360.0 * pi_2) * z_factor, math.sin(
                            angle_x[camid_horiz] / 360.0 *
                            pi_2) * z_factor, math.sin(angle_y[camid_vert] /
                                                       360.0 * pi_2)

                    # reposition the camera towards the target
                    camera.location = mathutils.Vector((cam_x, cam_y, cam_z)) * 6.0
                    camera = update_camera(camera, distance=dist_all[dist])

                    # get the camera intrinsic and extrinsic
                    P, K, RT, R, T = get_3x4_P_matrix_from_blender(camera)
                    intri, extri = write_cam_params(intri=intri,
                                                    extri=extri,
                                                    K=K,
                                                    RT=RT,
                                                    R=R,
                                                    T=T,
                                                    cam_loc=camera.location,
                                                    cam_tag=cam_tag)

                    dg = bpy.context.evaluated_depsgraph_get()
                    dg.update()

                    if save_videos:
                        bpy.data.scenes[
                            "Scene"].render.image_settings.file_format = "FFMPEG"
                        bpy.context.scene.render.filepath = os.path.join(
                            home_dir, 'Documents/datasets/videos',
                            str(cam_tag) + '.mkv')
                        bpy.ops.render.render(animation=True)

                    if save_images:
                        # Save RGB images as PNG
                        output_mask = setup_depth_node(home_dir, format='png')
                        """
                        output_mask.base_path = os.path.join(
                            home_dir, 'Documents/datasets/mask',
                            str(cam_tag)) + '/'
                        """
                        bpy.context.scene.render.filepath = os.path.join(
                            home_dir, 'Documents/datasets/images',
                            str(cam_tag))  + '/'
                        bpy.ops.render.render(animation=True)
                    elif save_depth:
                        # Save real depth images (mm) as EXR
                        output_depth = setup_depth_node(home_dir, format='exr')
                        output_depth.base_path = os.path.join(
                            home_dir, 'Documents/datasets/depth',
                            str(cam_tag)) + '/'
                        bpy.ops.render.render(animation=True)
                else:
                    z_factor = math.cos(angle_y[camid_vert] / 360.0 * pi_2)
                    cam_x, cam_y, cam_z = math.cos(
                        (angle_x[0] - interval_x) / 360.0 *
                        pi_2) * z_factor, math.sin(
                            (angle_x[0] - interval_x) / 360.0 *
                            pi_2) * z_factor, math.sin(angle_y[camid_vert] /
                                                       360.0 * pi_2)

                    # reposition the camera towards the target
                    camera.location = mathutils.Vector((cam_x, cam_y, cam_z))
                    # reposition the camera towards the target
                    camera = update_camera(camera, distance=dist_all[dist])
                    # get the camera intrinsic and extrinsic
                    P, K, RT, R, T = get_3x4_P_matrix_from_blender(camera)
                    dg = bpy.context.evaluated_depsgraph_get()
                    dg.update()

                    cam_x, cam_y, cam_z = math.cos(
                        (angle_x[0] - interval_x) / 360.0 *
                        pi_2) * z_factor, math.sin(
                            (angle_x[0] - interval_x) / 360.0 *
                            pi_2) * z_factor, math.sin(angle_y[camid_vert] /
                                                       360.0 * pi_2)

                    # reposition the camera towards the target
                    camera.location = mathutils.Vector((cam_x, cam_y, cam_z))
                    # reposition the camera towards the target
                    camera = update_camera(camera)
                    # get the camera intrinsic and extrinsic
                    P, K, RT, R, T = get_3x4_P_matrix_from_blender(camera)
                    dg = bpy.context.evaluated_depsgraph_get()
                    dg.update()

    # save intrinsics and extrinsics
    fname_in = os.path.join(home_dir, 'Documents/datasets/intri.json')
    fname_ex = os.path.join(home_dir, 'Documents/datasets/extri.json')
    with open(fname_in, "w") as f:
        json.dump(intri, f)
    with open(fname_ex, "w") as f:
        json.dump(extri, f)

    # delete all objects, e.g. lights and camera
    delete_objects()
