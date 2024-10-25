import os, torch, json, argparse, shutil
from predator.datasets.dataloader import get_dataloader, get_datasets
from easydict import EasyDict as edict
from predator.lib.utils import setup_seed, load_config
# ros include stuff
import rclpy
from rclpy.node import Node
import numpy as np
from fsregistration.srv import RequestListPotentialSolution3D
# from std_msgs.msg import Float64MultiArray
import open3d as o3d
import copy
import transforms3d.quaternions as quat


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(RequestListPotentialSolution3D, 'fs3D/registration/all_solutions')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestListPotentialSolution3D.Request()

    def send_request(self, scan1, scan2, N, VoxelSize):
        self.req.size_of_voxel = VoxelSize
        self.req.potential_for_necessary_peak = 0.1
        self.req.debug = False
        self.req.dimension_size = N
        self.req.sonar_scan_1 = scan1.tolist()
        self.req.sonar_scan_2 = scan2.tolist()
        self.req.timing_computation_duration = False
        self.req.use_clahe = True

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices(pcd1_down, pcd0_down, np.linalg.inv(trans),
                                      voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def getVoxelIndex(x, y, z, voxelSize, N):
    voxelX = int(x / voxelSize)
    voxelY = int(y / voxelSize)
    voxelZ = int(z / voxelSize)

    return int(voxelX + N / 2 + (voxelY + N / 2) * N + (voxelZ + N / 2) * N * N)


def pointToVoxel(pointcloud, N, voxelSize, shift):
    # voxelGrid = Float64MultiArray.data
    voxelGrid = np.zeros(N * N * N)

    # print(len(pointcloud.points))
    # print(pointcloud.points[0])
    # print(pointcloud.points[0][0])
    # print(pointcloud.points[1][1])
    # print(pointcloud.points[1][2])
    for point in pointcloud.points:
        pointShifted = point - shift
        index = getVoxelIndex(pointShifted[0], pointShifted[1], pointShifted[2], voxelSize, N);
        if index >= 0 and index < len(voxelGrid):
            voxelGrid[index] = 1
        # print(point[0], point[1], point[2])
        # print(index)

    # getVoxelIndex(x, y, z, voxelSize, N);
    return voxelGrid


def get_matching_indices(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    # source.transform(trans)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds


architectures = dict()
architectures['indoor'] = [
    'simple',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'resnetb_strided',
    'resnetb',
    'resnetb',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'unary',
    'nearest_upsample',
    'last_unary'
]

if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to the config file.')
    args = parser.parse_args()
    config = load_config(args.config)
    config['snapshot_dir'] = '%s' % config['exp_dir']
    config['tboard_dir'] = '%s/tensorboard' % config['exp_dir']
    config['save_dir'] = '%s/checkpoints' % config['exp_dir']
    config = edict(config)

    config.architecture = architectures[config.dataset]
    train_set, val_set, benchmark_set = get_datasets(config)
    config.train_loader, neighborhood_limits = get_dataloader(dataset=train_set,
                                                              batch_size=config.batch_size,
                                                              shuffle=False,
                                                              num_workers=config.num_workers,
                                                              )

    # ROS2 Node
    rclpy.init()

    minimal_client = MinimalClientAsync()

    # config.val_loader, _ = get_dataloader(dataset=val_set,
    #                                        batch_size=config.batch_size,
    #                                       shuffle=False,
    #                                       num_workers=0,
    #                                       neighborhood_limits=neighborhood_limits
    #                                       )
    # config.test_loader, _ = get_dataloader(dataset=benchmark_set,
    #                                        batch_size=config.batch_size,
    #                                        shuffle=False,
    #                                        num_workers=0,
    #                                        neighborhood_limits=neighborhood_limits)

    for _ in range(10):
        inputs = next(iter(config.train_loader))

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
        pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])

        mean1, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd1)
        mean2, _ = o3d.geometry.PointCloud.compute_mean_and_covariance(pcd2)
        print(mean1)
        print(mean2)
        # draw_registration_result(pcd1,pcd2,np.identity(4))

        T = np.eye(4)
        T[:3, :3] = inputs["rot"]
        T[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))
        print("rotation Matrix GT: ")
        print(T)
        print("rotation Quat GT: ")
        print(quat.mat2quat(T[0:3,0:3]))
        voxelSize = 0.05
        print("percentage Overlap: ", compute_overlap_ratio(pcd1, pcd2, T, voxelSize))
        pcd1Vox = pcd1.voxel_down_sample(voxel_size=voxelSize)
        pcd2Vox = pcd2.voxel_down_sample(voxel_size=voxelSize)
        N = 128
        voxelSize = 0.05
        voxelArray1 = pointToVoxel(pcd1, N, voxelSize, mean1).astype(np.float64)
        voxelArray2 = pointToVoxel(pcd2, N, voxelSize, mean2).astype(np.float64)

        response = RequestListPotentialSolution3D.Response()
        response = minimal_client.send_request(voxelArray1, voxelArray2, N, voxelSize)
        heightFirstPotentialSolution = response.list_potential_solutions[0].transformation_peak_height
        for peak in response.list_potential_solutions:
            currentQuaternion = [peak.resulting_transformation.orientation.w, peak.resulting_transformation.orientation.x,peak.resulting_transformation.orientation.y, peak.resulting_transformation.orientation.z]
            currentQuaternionInv = quat.qinverse(currentQuaternion)

            np.set_printoptions(suppress=True)
            print("peak: ", peak.transformation_peak_height / heightFirstPotentialSolution)
            print("x,y,z: ", peak.resulting_transformation.position)
            print("rotation1: ")
            print(o3d.geometry.get_rotation_matrix_from_quaternion(currentQuaternion))
            print(currentQuaternion)
            print("rotation2: ")
            print(o3d.geometry.get_rotation_matrix_from_quaternion(currentQuaternionInv))
            print(currentQuaternionInv)
            print("test")

        # print(response)
        draw_registration_result(pcd1Vox, pcd2Vox, T)

        print("test2")
