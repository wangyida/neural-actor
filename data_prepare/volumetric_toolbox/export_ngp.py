#!/usr/bin/env python3
# Copyright (c) 2022 Synthesia Limited - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidental.

import argparse
import json
import math
import os
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R
from syna.experiments.volumetric_toolbox.camera_data import CameraData, read_calibration_csv


def export_as_ngp(
    cameras: List[CameraData],
    output_folder: Path,
    image_folder: Path,
    scene_offset: np.array,
    scene_scale: float,
    frame_index: int = 1,
    check_frame_exists: bool = True,
) -> None:
    frames = []

    to_ngp_camera = R.from_euler("x", [180], degrees=True).as_matrix()
    to_ngp_world = R.from_euler("xz", [90, 90], degrees=True).as_matrix()
    for camera_idx, camera in enumerate(cameras):
        matrix = np.eye(4)
        matrix[:3, :3] = to_ngp_world @ R.from_rotvec(camera.rotation_axisangle).as_matrix() @ to_ngp_camera
        matrix[:3, 3] = to_ngp_world @ ((camera.translation + scene_offset) * scene_scale)

        path_frame = (image_folder / (camera.name + f"_{frame_index:06d}.jpg"))
        if check_frame_exists and not path_frame.exists():
            path_frame_png = path_frame.with_suffix(".png")
            if path_frame_png.exists():
                path_frame = path_frame_png
            else:
                print("Skipping non-existing view:", path_frame)
                continue

        # For camera_angle_x, camera_angle_y see:
        # https://github.com/NVlabs/instant-ngp/blob/1dc8eb6318e47407c8296b0d9549c602280f39be/scripts/colmap2nerf.py#L216
        frames.append({
            "file_path": str(os.path.relpath(path_frame, output_folder)),
            "camera_name": camera.name,
            "transform_matrix": [list(v) for v in list(matrix)],
        })

        output = {
            "cx": camera.cx_pixel,
            "cy": camera.cy_pixel,
            "w": camera.width,
            "h": camera.height,
            "aabb_scale": 1,
            "frames": frames,
            "fl_x": camera.fx_pixel,
            "fl_y": camera.fy_pixel,
            "camera_angle_x": 2.0 * math.atan2(0.5 * camera.width, camera.fx_pixel),
            "camera_angle_y": 2.0 * math.atan2(0.5 * camera.height, camera.fy_pixel),
            "p1": 0.0,  # These are optional
            "p2": 0.0,  # These are optional
            "k1": 0.0,  # These are optional
            "k2": 0.0,  # These are optional
        }

        output_json_path = output_folder / f"transforms{camera_idx:03d}.json"
        output_json_path.write_text(json.dumps(output, indent=2), encoding="UTF-8")
        frames = []


def estimate_aabb_cameras(cameras: List[CameraData]) -> np.array:
    cam_locations = np.vstack([c.translation for c in cameras])
    return np.vstack([cam_locations.min(0), cam_locations.max(0)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--scene_bbox_ratio", type=float, default=0.95)
    args = parser.parse_args()

    cameras = read_calibration_csv(args.csv)
    aabb = estimate_aabb_cameras(cameras)

    # The NGP loaded scales the scene by 0.33, hence we scale the scene to ~95% * 3.
    # see: https://github.com/NVlabs/instant-ngp/blob/1dc8eb6318e47407c8296b0d9549c602280f39be/include/neural-graphics-primitives/nerf_loader.h#L28
    # and: https://github.com/NVlabs/instant-ngp/blob/master/docs/nerf_dataset_tips.md#existing-datasets
    scene_scale = args.scene_bbox_ratio * (3 / np.max(aabb[1] - aabb[0]))
    scene_offset = -aabb.mean(0)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_as_ngp(
        cameras,
        output_folder=args.output_dir,
        image_folder=args.image_folder,
        scene_offset=scene_offset,
        scene_scale=scene_scale
    )


if __name__ == "__main__":
    main()
