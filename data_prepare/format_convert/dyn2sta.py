import os
import shutil
import argparse
import numpy as np
from os.path import join
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--mode', type=str, default='static', choices=['static', 'dynamic'], help="static or dynamic")
    args = parser.parse_args()
    path = args.path
    mode = args.mode

    # NOTE: new codes
    """
    home_path = os.path.expanduser('~')
    path_img = os.path.join(home_path, 'Documents/datasets/images')
    path_mask = os.path.join(home_path, 'Documents/datasets/mask')
    """
    path_img = os.path.join(path, 'images')
    path_mask = os.path.join(path, 'mask')
    cameras_id = os.listdir(path_img)
    cameras_id = list(set(cameras_id) - set(['static']))
    target_path = os.path.join(path, mode)

    if not os.path.exists(target_path): 
        os.makedirs(target_path)
    if not os.path.exists(os.path.join(target_path, 'images')): 
        os.makedirs(os.path.join(target_path, 'images'))
    if not os.path.exists(os.path.join(target_path, 'mask')): 
        os.makedirs(os.path.join(target_path, 'mask'))

    if mode == 'static':
        for camera in cameras_id:
            shutil.copy(os.path.join(path_img, camera, '0001.png'), os.path.join(target_path, 'images', camera.zfill(4) + '.png'))
            shutil.copy(os.path.join(path_mask, camera, '0001.png'), os.path.join(target_path, 'mask', camera.zfill(4) + '.png'))
    elif mode == 'dynamic':
        for camera in cameras_id:
            shutil.copy(os.path.join(path_img, camera, camera.zfill(4) + '.png'), os.path.join(target_path, 'images', camera.zfill(4) + '.png'))
