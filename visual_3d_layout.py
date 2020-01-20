import numpy as np
from scipy.ndimage import map_coordinates

import pptk
from PIL import Image
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import functools
from multiprocessing import Pool

from utils_eval import np_coor2xy, np_coory2v


def xyz_2_coorxy(xs, ys, zs, H, W):
    us = np.arctan2(xs, -ys)
    vs = -np.arctan(zs / np.sqrt(xs**2 + ys**2))
    coorx = (us / (2 * np.pi) + 0.5) * W
    coory = (vs / np.pi + 0.5) * H
    return coorx, coory


def pt_in_poly(poly, pt):
    return poly.contains(Point(pt))


def warp_walls(xy, floor_z, ceil_z, H, W, ppm, alpha):
    all_rgba = []
    all_xyz = []
    for i in range(len(xy)):
        next_i = (i + 1) % len(xy)
        xy_a = xy[i]
        xy_b = xy[next_i]
        xy_w = np.sqrt(((xy_a - xy_b)**2).sum())
        t_h = int(round((ceil_z - floor_z) * ppm))
        t_w = int(round(xy_w * ppm))
        xs = np.linspace(xy_a[0], xy_b[0], t_w)[None].repeat(t_h, 0)
        ys = np.linspace(xy_a[1], xy_b[1], t_w)[None].repeat(t_h, 0)
        zs = np.linspace(floor_z, ceil_z, t_h)[:, None].repeat(t_w, 1)
        coorx, coory = xyz_2_coorxy(xs, ys, zs, H, W)

        plane_texture = np.stack([
            map_coordinates(equirect_texture[..., 0], [coory, coorx], order=1, mode='wrap'),
            map_coordinates(equirect_texture[..., 1], [coory, coorx], order=1, mode='wrap'),
            map_coordinates(equirect_texture[..., 2], [coory, coorx], order=1, mode='wrap'),
            np.zeros([t_h, t_w]) + alpha,
        ], -1)
        plane_xyz = np.stack([xs, ys, zs], axis=-1)

        all_rgba.extend(plane_texture.reshape(-1, 4))
        all_xyz.extend(plane_xyz.reshape(-1, 3))

    return all_rgba, all_xyz


def warp_floor_ceiling(xy, z_floor, z_ceiling, H, W, ppm, alpha, n_thread):
    min_x = xy[:, 0].min()
    max_x = xy[:, 0].max()
    min_y = xy[:, 1].min()
    max_y = xy[:, 1].max()
    t_h = int(round((max_y - min_y) * ppm))
    t_w = int(round((max_x - min_x) * ppm))
    xs = np.linspace(min_x, max_x, t_w)[None].repeat(t_h, 0)
    ys = np.linspace(min_y, max_y, t_h)[:, None].repeat(t_w, 1)
    zs_floor = np.zeros_like(xs) + z_floor
    zs_ceil = np.zeros_like(xs) + z_ceiling
    coorx_floor, coory_floor = xyz_2_coorxy(xs, ys, zs_floor, H, W)
    coorx_ceil, coory_ceil = xyz_2_coorxy(xs, ys, zs_ceil, H, W)

    floor_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_floor, coorx_floor], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_floor, coorx_floor], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1).reshape(-1, 4)
    floor_xyz = np.stack([xs, ys, zs_floor], axis=-1).reshape(-1, 3)

    ceil_texture = np.stack([
        map_coordinates(equirect_texture[..., 0], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 1], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        map_coordinates(equirect_texture[..., 2], [coory_ceil, coorx_ceil], order=1, mode='wrap'),
        np.zeros([t_h, t_w]) + alpha,
    ], -1).reshape(-1, 4)
    ceil_xyz = np.stack([xs, ys, zs_ceil], axis=-1).reshape(-1, 3)

    xy_poly = Polygon(xy)
    with Pool(n_thread) as p:
        sel = p.map(functools.partial(pt_in_poly, xy_poly), floor_xyz[:, :2])

    return floor_texture[sel], floor_xyz[sel], ceil_texture[sel], ceil_xyz[sel]


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--img', default='assert/output_preprocess/demo_aligned_rgb.png',
                        help='Image texture in equirectangular format')
    parser.add_argument('--layout', default='assert/output/demo_aligned_rgb_cor_id.txt',
                        help='Txt file containing layout corners (cor_id)')
    parser.add_argument('--camera_height', default=1.6, type=float,
                        help='Camera height in meter (not the viewer camera)')
    parser.add_argument('--ppm', default=150, type=int,
                        help='Points per meter')
    parser.add_argument('--point_size', default=0.0025, type=int,
                        help='Point size')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='Opacity of the texture')
    parser.add_argument('--threads', default=10, type=int,
                        help='Number of threads to use')
    parser.add_argument('--ignore_floor', action='store_true',
                        help='Skip rendering floor')
    parser.add_argument('--ignore_ceiling', action='store_true',
                        help='Skip rendering ceiling')
    args = parser.parse_args()

    # Reading source (texture img, cor_id txt)
    equirect_texture = np.array(Image.open(args.img)) / 255.0
    with open(args.layout) as f:
        cor_id = np.array([line.split() for line in f], np.float32)

    # Convert cor_id to 3d xyz
    N = len(cor_id) // 2
    H, W = equirect_texture.shape[:2]
    floor_z = -args.camera_height
    floor_xy = np_coor2xy(cor_id[1::2], floor_z, W, H)
    c = np.sqrt((floor_xy**2).sum(1))
    v = np_coory2v(cor_id[0::2, 1], H)
    ceil_z = (c * np.tan(v)).mean()

    # Warp each wall
    all_rgba, all_xyz = warp_walls(floor_xy, floor_z, ceil_z, H, W, args.ppm, args.alpha)

    # Warp floor and ceiling
    if not args.ignore_floor or not args.ignore_ceiling:
        fi, fp, ci, cp = warp_floor_ceiling(floor_xy, floor_z, ceil_z, H, W,
                                            ppm=args.ppm,
                                            alpha=args.alpha,
                                            n_thread=args.threads)

        if not args.ignore_floor:
            all_rgba.extend(fi)
            all_xyz.extend(fp)

        if not args.ignore_ceiling:
            all_rgba.extend(ci)
            all_xyz.extend(cp)

    # Launch point cloud viewer
    print('# of points:', len(all_rgba))
    viewer = pptk.viewer(all_xyz, all_rgba)
    viewer.set(point_size=args.point_size, lookat=[0, 0, 0])
