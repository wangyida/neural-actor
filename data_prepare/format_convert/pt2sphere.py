#from urllib.robotparser import RequestRate
import open3d as o3d
import argparse
import numpy as np
from matplotlib import pyplot as plt


def o3d_pts2mesh(pc: o3d.geometry.PointCloud,
                 radius: float = 1e-3,
                 resolution: int = 10):
    pc = pc.voxel_down_sample(voxel_size=0.02)
    pts = np.asarray(pc.points)
    clrs = np.asarray(pc.colors)
    # radius = np.asarray(pc.compute_nearest_neighbor_distance()).mean()
    spheres = [
        o3d.geometry.TriangleMesh.create_sphere(radius, resolution)
        for _ in range(len(pts))
    ]

    output = o3d.geometry.TriangleMesh()
    for idx, pt in enumerate(pts):
        spheres[idx].translate(pt)
        spheres[idx].paint_uniform_color(clrs[idx])
        output += spheres[idx]
    output.compute_vertex_normals()
    # o3d.visualization.draw_geometries([output])
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        help='input filename',
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        help='output filename',
                        required=True)
    parser.add_argument('-re',
                        '--resolution',
                        type=int,
                        default=4,
                        help='sphere resolution',
                        required=False)
    parser.add_argument('-ra',
                        '--radius',
                        type=float,
                        default=1e-2,
                        help='sphere radius',
                        required=False)
    opt = args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(opt.input)
    mesh = o3d_pts2mesh(pcd, opt.radius, opt.resolution)
    o3d.io.write_triangle_mesh(opt.output,
                               mesh,
                               write_ascii=False,
                               compressed=True)
