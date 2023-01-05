#!/usr/bin/env python3
import os
import numpy as np
from camera_data import CameraData, read_calibration_csv
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1,
        help='camera id')
    args = parser.parse_args()
home_dir = os.path.expanduser('~')
fname = os.path.join(home_dir, "Documents/datasets_synthesia_%s" % (args.id))
cameras = read_calibration_csv(fname)
