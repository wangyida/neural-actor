import open3d as o3d
import os
from os.path import join
import json
from tqdm import tqdm
import numpy as np
import cv2


def read_rgbd(image_root, depth_root, with_noise=False):
    color = cv2.imread(image_root, cv2.IMREAD_UNCHANGED)
    """
    cv2.imshow('sds', color)
    cv2.waitKey(0)
    """
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    depth = cv2.imread(depth_root, cv2.IMREAD_UNCHANGED)

    # Noise test
    if with_noise:
        row, col, ch = depth.shape
        mean = 0
        var = 0.0002
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        gauss = gauss.astype(np.float32)
        depth = depth + gauss
    return color, depth


def depth2pcd(intri, extri, path, vis=False, ext='png'):
    pointSet = o3d.geometry.PointCloud()
    os.makedirs(join(path, 'pcd'), exist_ok=True)
    # Extrinsics
    f_extri = open(join(path, extri + '.json'), 'r')
    json_ex = json.load(f_extri)
    f_extri.close()
    for i in tqdm(range(0, len(json_ex), 3)):
        name_idx = i // 3 + 1
        if i >= 6:
            name_idx = name_idx + 1
        # Intrinsics
        # Data to be written
        # NOTE as posted in https://github.com/yenchenlin/nerf-pytorch/issues/41
        # focal = 0.5 * image_width / np.tan(0.5 * camera_angle_x)
        f_intri = open(join(path, intri + '.json'), 'r')
        json_in = json.load(f_intri)
        f_intri.close()
        """
        # The matrix looks like
        f_x  s    c_x
        0    f_y  c_y
        0    0    1
        """
        fx = json_in[i//3*2+1]['K_' + str(name_idx)]['data'][0]
        fy = json_in[i//3*2+1]['K_' + str(name_idx)]['data'][4]
        cx = json_in[i//3*2+1]['K_' + str(name_idx)]['data'][2]
        cy = json_in[i//3*2+1]['K_' + str(name_idx)]['data'][5]

        array4x4 = []
        temp = json_ex[i]['Rot_' + str(name_idx)]['data'][0:3]
        temp.append(json_ex[i + 1]['T_' + str(name_idx)]['data'][0])
        array4x4.append(temp)
        temp = json_ex[i]['Rot_' + str(name_idx)]['data'][3:6]
        temp.append(json_ex[i + 1]['T_' + str(name_idx)]['data'][1])
        array4x4.append(temp)
        temp = json_ex[i]['Rot_' + str(name_idx)]['data'][6:9]
        temp.append(json_ex[i + 1]['T_' + str(name_idx)]['data'][2])
        array4x4.append(temp)
        array4x4.append([0.0, 0.0, 0.0, 1.0])
        arr_rt = np.array(array4x4)

        frame_id = '0001'
        """
        if i == 0:
            with_noise = False
            rgb, depth = read_rgbd(image_root=join(path, 'images',
                                                   str(name_idx),
                                                   frame_id + '.png'),
                                   depth_root=join(path, 'depth', str(name_idx),
                                                   'Image' + frame_id + '.exr'), with_noise=with_noise)
            print(np.max(depth), np.min(depth))
        else:
        """
        with_noise = False
        rgb, depth = read_rgbd(image_root=join(path, 'images',
                                               str(name_idx),
                                               frame_id + '.' + ext),
                               depth_root=join(path, 'depth', str(name_idx),
                                               frame_id + '.exr'), with_noise=with_noise)
        image_width = int(depth.shape[1])
        image_height = int(depth.shape[0])
        camera_angle_x = np.arctan((0.5 * image_width) / fx) / 0.5
        print(np.max(depth), np.min(depth))
        img = o3d.geometry.Image(rgb)
        dep_max = 10.0
        depth_o3d = o3d.geometry.Image(depth)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            img,
            depth_o3d,
            depth_trunc=dep_max - 0.01,
            depth_scale=1.0,
            convert_rgb_to_intensity=False)

        print('Image', name_idx, image_width, image_height, fx, fy, cx, cy)
        o3dintrinsic = o3d.camera.PinholeCameraIntrinsic(
            image_width, image_height, fx, fy, cx, cy)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, intrinsic=o3dintrinsic, extrinsic=arr_rt)

        # save point cloud
        os.makedirs(join(path, 'pcd', str(name_idx)), exist_ok=True)
        o3d.io.write_point_cloud(join(path, 'pcd', str(name_idx),
                                      frame_id + '.ply'),
                                 pcd,
                                 write_ascii=True)

        # NOTE a direct back-projection
        xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        if depth.shape[-1] == 3:
            depth = depth[:,:,0]
        coords = np.stack([xx, yy, np.ones_like(xx)], axis=-1) * depth[:, :, np.newaxis]
        print(coords.shape)
        coords = coords[depth > 0].reshape(-1, 3)
        # coords = coords[depth < 10].reshape(-1, 3)
        Kinv = np.linalg.inv(o3dintrinsic.intrinsic_matrix)
        # pcd_mustafa = cam.rotation_matrix() @ Kinv @ coords.T + cam.translation[:, None]
        points_tmp = arr_rt[:3,:3] @ Kinv @ coords.T + arr_rt[:3,3, None]
        pcd_mus = o3d.geometry.PointCloud()
        pcd_mus.points = o3d.utility.Vector3dVector(points_tmp.T)

        if vis:
            o3d.visualization.draw_geometries([pcd_mus],
                                              zoom=0.45,
                                              front=[ 0.042, -0.950, 0.307],
                                              lookat=[ -0.140, 0.063, -0.027],
                                              up=[ 0.011, 0.308, 0.951])

        pointSet.points = (o3d.utility.Vector3dVector(
            np.concatenate((np.array(pointSet.points), np.array(pcd.points)),
                           axis=0)))
        pointSet.colors = (o3d.utility.Vector3dVector(
            np.concatenate((np.array(pointSet.colors), np.array(pcd.colors)),
                           axis=0)))
        pointSet = pointSet.voxel_down_sample(voxel_size=0.008)
        o3d.io.write_point_cloud('./whole_body.ply',
                                 pointSet,
                                 write_ascii=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--ext',
                        type=str,
                        default='jpg',
                        choices=['jpg', 'png'],
                        help="image file extension")
    parser.add_argument('--vis',
                        action='store_true',
                        help="visualize point cloud")
    args = parser.parse_args()

    depth2pcd('intri', 'extri', path=args.path, vis=args.vis, ext=args.ext)
