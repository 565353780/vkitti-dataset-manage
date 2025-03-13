import os
import cv2
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm
from typing import Union


def loadDepthImage(depth_file_path: str) -> Union[np.ndarray, None]:
    if not os.path.exists(depth_file_path):
        print('[ERROR][demo::loadDepthImage]')
        print('\t depth file not exist!')
        print('\t depth_file_path:', depth_file_path)
        return None

    depth = cv2.imread(depth_file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) / 100.0
    return depth

def get_rotation_matrix(rot_x: np.ndarray, rot_y: np.ndarray, rot_z: np.ndarray) -> np.ndarray:
    """
    create 3D rotation matrix given rotations around axes in rad
    :param rot_x: rotation around x in rad
    :param rot_y: rotation around y in rad
    :param rot_z: rotation around z in rad
    :return: 3D rotation matrix
    """
    c, s = np.cos(rot_z), np.sin(rot_z)

    RZ = np.matrix(
        '{} {} 0;'
        '{} {} 0;'
        ' 0  0 1'.format(c, -s, s, c)
    )

    c, s = np.cos(rot_x), np.sin(rot_x)

    RX = np.matrix(
        ' 1  0  0;'
        ' 0 {} {};'
        ' 0 {} {}'.format(c, -s, s, c)
    )

    c, s = np.cos(rot_y), np.sin(rot_y)

    RY = np.matrix(
        '{} {}  0;'
        '{} {}  0;'
        ' 0  0  1'.format(c, -s, s, c)
    )

    return RX.dot(RY).dot(RZ)

def distance_cutoff(pointcloud: np.ndarray, cutoff: float = 100.) -> np.ndarray:
    """
    points which are too far away are clipped away
    :param pointcloud: point cloud
    :param cutoff: maximal distance for considering a point
    :return: clipped point cloud
    """
    cut = np.sqrt(np.square(pointcloud[:, :3]).sum(axis=1)) < cutoff
    return pointcloud[cut]

def transform2worldspace(pointcloud: np.ndarray, mf) -> np.ndarray:
    """
    transform from camera space into world space
    :param pointcloud: point cloud
    :param img_no: image number within sequence
    :return: point cloud in world space
    """
    worldspace = pointcloud.copy()
    worldspace = worldspace[:, (1,2,0)]
    worldspace[:, 0] = -worldspace[:,0]
    worldspace[:, 1] = -worldspace[:,1]
    worldspace[:, 2] = worldspace[:,2]

    translation = np.array([float(mf['t1']), float(mf['t2']), float(mf['t3'])])

    matrix = np.array([[float(mf['r1,1']), float(mf['r1,2']), float(mf['r1,3'])],
                       [float(mf['r2,1']), float(mf['r2,2']), float(mf['r2,3'])],
                       [float(mf['r3,1']), float(mf['r3,2']), float(mf['r3,3'])]])

    # set the coordinate system origin to the center of the vehicle
    worldspace[:, :3] = worldspace[:, :3] - translation

    # rotate all points such that the vehicle is axis-aligned
    worldspace[:, :3] = np.einsum('ij, kj -> ki', np.transpose(matrix), worldspace[:, :3])

    return worldspace

def toPcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def toMergedPcd(pcd_list: list) -> o3d.geometry.PointCloud:
    merged_pc = o3d.geometry.PointCloud()

    for pcd in pcd_list:
        merged_pc += pcd

    return merged_pc

if __name__ == '__main__':
    center_x = 620.5
    center_y = 187.0
    focal_x = 725.0
    focal_y = 725.0

    yv, xv = np.meshgrid(range(375), range(1242), indexing='ij')

    g_cutoff = 100.

    ci, cj, fi, fj = 187, 620.5, 725.0087, 725.0087

    camera_id = 0

    scene_dir = '/home/chli/chLi/Dataset/VirtualKITTI/vkitti_2.0.3_depth/Scene01/clone/frames/depth/Camera_' + str(camera_id) + '/'
    ext_file_path = '/home/chli/chLi/Dataset/VirtualKITTI/vkitti_2.0.3_textgt/Scene01/clone/extrinsic.txt'

    merge_depth_num = 500
    down_sample_ratio = 0.001

    extgt = pd.read_csv(ext_file_path, sep=' ')

    depth_file_list = os.listdir(scene_dir)

    pcd_list = []

    for depth_file in tqdm(depth_file_list):
        if not depth_file.endswith('.png'):
            continue

        depth_id = int(depth_file.split('.')[0].split('_')[1])

        mf = extgt[(extgt['frame'] == depth_id) & (extgt['cameraID'] == camera_id)]

        depth_file_path = scene_dir + depth_file
        assert os.path.exists(depth_file_path)

        depth_map = loadDepthImage(depth_file_path)
        assert depth_map is not None

        x3 = (xv - center_x) / focal_x * depth_map
        y3 = (yv - center_y) / focal_y * depth_map

        erg = np.stack((depth_map, -x3, -y3), axis=-1).reshape((-1, 3))

        # delete sky points
        erg = distance_cutoff(erg, g_cutoff)

        # erg = remove_car_shadows(erg, img_no, g_bb_eps)
        worldspace = transform2worldspace(erg, mf)

        pcd = toPcd(worldspace)

        pcd = pcd.random_down_sample(down_sample_ratio)

        pcd_list.append(pcd)

    merged_pcd = toMergedPcd(pcd_list)

    o3d.io.write_point_cloud('./merged_pcd_downsample-' + str(down_sample_ratio).replace('.', '-') + '.ply', merged_pcd, write_ascii=True)

    #o3d.visualization.draw_geometries([merged_pcd])
