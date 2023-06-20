import os, sys
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob
import numpy as np

mkdir = lambda x: os.makedirs(x, exist_ok=True)


def convert_rgbd(image_root, depth_root, mask_root, ext='jpg'):
    imgnames = sorted(os.listdir(image_root))
    for imgname in imgnames:
        if imgname.endswith(ext):
            im = cv2.imread(join(image_root, imgname), cv2.IMREAD_UNCHANGED)
            # depth = cv2.imread(join(depth_root, "Image"+imgname.replace(ext, "exr")), cv2.IMREAD_UNCHANGED)
            depth = cv2.imread(join(depth_root, imgname.replace(ext, "exr")), cv2.IMREAD_UNCHANGED)
            # im[:,:,3] = depth[:,:,0]
            # Setup a margine of 10 meters to trim off background
            im[:,:,3][depth[:,:,0] >= 35.0] = 0
            im[:,:,3][depth[:,:,0] < 35.0] = 255
            cv2.imwrite(join(image_root, imgname.replace(ext, "png")), im)
            # cv2.imwrite(join(depth_root, imgname), depth[:,:,0])
            # cv2.imwrite(join(depth_root, imgname.replace(ext, "png")), im[:,:,3])
            # cv2.imwrite(join(mask_root, imgname.replace(ext, "png")), im[:,:,3] / 255)
            cv2.imwrite(join(mask_root, imgname.replace(ext, "png")), im[:,:,3])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--ext', type=str, default='png', choices=['jpg', 'png', 'exr'], help="image file extension")
    args = parser.parse_args()

    os.makedirs(join(args.path, 'mask'), exist_ok=True)
    if os.path.isdir(args.path):
        image_path = join(args.path, 'images')
        subs = sorted(os.listdir(image_path))
        for sub in tqdm(subs):
            image_root = join(args.path, 'images', sub)
            depth_root = join(args.path, 'depth', sub)
            mask_root = join(args.path, 'mask', sub)
            os.makedirs(join(args.path, 'mask', sub), exist_ok=True)
            convert_rgbd(image_root, depth_root, mask_root, ext=args.ext)
    else:
        print(args.path, ' not exists')
