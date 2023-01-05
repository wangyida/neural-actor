# Copyright (c) 2022 Synthesia Limited - All Rights Reserved
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidental.

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from scipy.spatial.transform import Rotation as R

def write_cam_params(intri, extri, K, RT, R, T, cam_tag):
    # Make them into lists
    K = [x for xs in K for x in xs]
    R_from_RT = RT[:3,:3]
    RT = [x for xs in RT for x in xs]
    # R = [x for xs in R_from_RT for x in xs]
    R = [x for xs in R for x in xs]
    T = [x for x in T]
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
            "data": [0,0,0],
            "rows": 3,
            "cols": 1,
            "dt": 'd'
        }
    })
    return intri, extri

def write_cam_params_txt(intri, extri, K, RT, R, T, cam_tag):
    # Make them into lists
    K = [x for xs in K for x in xs]
    RT = [x for xs in RT for x in xs]
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
            "data": [0,0,0],
            "rows": 3,
            "cols": 1,
            "dt": 'd'
        }
    })
    return intri, extri

@dataclass
class CameraData:
    """
    Camera coordinate system uses right-down-forward (RDF) convention similarly to COLMAP.

    Extrinsics represent the transformation from camera-space to world-space.
    * The magnitude of `rotation_axisangle` defines the rotation angle in radians.
    * `translation` is typically stored in meters [m].
    """
    name: str
    width: int
    height: int

    # Extrinsics
    rotation_axisangle: np.array
    translation: np.array

    # Intrinsics
    focal_length: np.array
    principal_point: np.array

    # Optional distortion coefficients
    k1: float = 0
    k2: float = 0
    k3: float = 0

    @property
    def fx_pixel(self):
        return self.width * self.focal_length[0]

    @property
    def fy_pixel(self):
        return self.height * self.focal_length[1]

    @property
    def cx_pixel(self):
        return self.width * self.principal_point[0]

    @property
    def cy_pixel(self):
        return self.height * self.principal_point[1]

    def intrinsic_matrix(self, width, height, downsample_rate=1.0):
        factor = 1.0 / downsample_rate
        if width > 100000000: #height:
            return np.array([
                [self.fy_pixel * factor, 0, (height - self.cy_pixel) * factor],
                [0, self.fx_pixel * factor, self.cx_pixel * factor],
                [0, 0, 1],
            ])
        else:
            return np.array([
                [self.fx_pixel * factor, 0, self.cx_pixel * factor],
                [0, self.fy_pixel * factor, self.cy_pixel * factor],
                [0, 0, 1],
            ])

    def rotation_matrix(self) -> np.array:
        """Rotation matrix of the camera to world transform. Matrix should be applied from the left, points are
        assumed to be represented as columns.
        e.g. R @ points

        Returns:
            np.array: Rotation matrix (3 x 3)
        """
        return R.from_rotvec(self.rotation_axisangle).as_matrix()

    def extrinsic_matrix_cam2world(self, width, height) -> np.array:
        """Set up camera to world transformation matrix to be left-multiplied to points in homogenous coordinates
        e.g. R @ points

        Returns:
            np.array (4 x 4): Transformation matrix going from camera view to world coordinate system
        """
        mat = np.eye(4)
        mat[:3, :3] = self.rotation_matrix()
        mat[:3, 3] = self.translation
        return mat

    def extrinsic_matrix_world2cam(self, width, height) -> np.array:
        """Set up camera to world transformation matrix to be left-multiplied to points in homogenous coordinates
        e.g. R @ points

        Returns:
            np.array (4 x 4): Transformation matrix going from world coordinate system to camera view
        """
        mat = np.eye(4)
        mat[:3, :3] = self.rotation_matrix()
        # if width > height:
        if width > 100000000: #height:
            cam_rot_90 = np.zeros((3, 3))
            cam_rot_90[0, 1] = -1.0
            cam_rot_90[1, 0] = 1.0
            cam_rot_90[2, 2] = 1.0
            mat[:3, :3] = np.matmul(cam_rot_90, mat[:3, :3])
        mat[:3, 3] = self.translation
        mat = np.linalg.inv(mat)
        return mat

def write_calibration_csv(
    cameras: List[CameraData],
    output_csv_path: Path
) -> None:
    """Write camera intrinsics and extrinsics to a calibration CSV file.

    Args:
        cameras (List[CameraData]): List `CameraData` objects describing camera parameters.
        output_csv_path (Path): Path of output CSV file.
    """
    csv_field_names = [
        "name", "w", "h", "rx", "ry", "rz", "tx", "ty", "tz", "fx", "fy", "px", "py"
    ]
    with open(output_csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_field_names)
        writer.writeheader()

        for camera in cameras:
            csv_row = {}
            csv_row["name"] = camera.name
            csv_row["w"] = camera.width
            csv_row["h"] = camera.height
            csv_row["rx"] = camera.rotation_axisangle[0]
            csv_row["ry"] = camera.rotation_axisangle[1]
            csv_row["rz"] = camera.rotation_axisangle[2]
            csv_row["tx"] = camera.translation[0]
            csv_row["ty"] = camera.translation[1]
            csv_row["tz"] = camera.translation[2]
            csv_row["fx"] = camera.focal_length[0]
            csv_row["fy"] = camera.focal_length[1]
            csv_row["px"] = camera.principal_point[0]
            csv_row["py"] = camera.principal_point[1]

            assert len(csv_row) == len(csv_field_names)
            writer.writerow(csv_row)


def read_calibration_csv(
    input_csv_path: Path
) -> List[CameraData]:
    """Read camera intrinsics and extrinsics from a calibration CSV file.

    Args:
        input_csv_path (Path): Path to CSV file that contains camera calibration data.

    Returns:
        List[CameraData]: A list of `CameraData` objects that describes multiple camera
                          intrinsics and extrinsics.
    """
    temp = []
    cameras = []
    intri = []
    extri = []
    # 160 cameras in total
    name_indexes = list(range(1, 160 + 1, 1))
    cam_tags = [str(x) for x in name_indexes]
    intri.append({"names": cam_tags})
    with open(os.path.join(input_csv_path, 'calibration.csv'), "r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            camera = CameraData(
                name=row["name"],
                width=int(row["w"]),
                height=int(row["h"]),
                rotation_axisangle=np.array([float(row["rx"]), float(row["ry"]), float(row["rz"])]),
                translation=np.array([float(row["tx"]), float(row["ty"]), float(row["tz"])]),
                focal_length=np.array([float(row["fx"]), float(row["fy"])]),
                principal_point=np.array([float(row["px"]), float(row["py"])]),
            )
            cameras.append(camera)

            # NOTE: write camera files
            cam_id = int(row["name"][-3:])
            intri, extri = write_cam_params(intri=intri,
                                            extri=extri,
                                            K=camera.intrinsic_matrix(int(row["w"]), int(row["h"]), downsample_rate=1.0),
                                            RT=camera.extrinsic_matrix_world2cam(int(row["w"]), int(row["h"])),
                                            R=camera.extrinsic_matrix_world2cam(int(row["w"]), int(row["h"]))[:3,:3],
                                            T=camera.extrinsic_matrix_world2cam(int(row["w"]), int(row["h"]))[:3,3],
                                            cam_tag=str(cam_id))

            """
            with open("yes.txt", "ab") as f:
                f.write(b"\n")
                np.savetxt(f, np.array(camera.intrinsic_matrix()))
            with open("yes.txt", "ab") as f:
                np.savetxt(f, np.array([0.0])) # np.zeros(shape = (1)))
            with open("yes.txt", "ab") as f:
                np.savetxt(f, np.array(np.linalg.inv(camera.extrinsic_matrix_cam2world()))[:3, :])
            """

    # save intrinsics and extrinsics
    home_dir = os.path.expanduser('~')
    fname_in = os.path.join(input_csv_path, 'intri.json')
    fname_ex = os.path.join(input_csv_path, 'extri.json')
    with open(fname_in, "w") as f:
        json.dump(intri, f)
    with open(fname_ex, "w") as f:
        json.dump(extri, f)
    return cameras
