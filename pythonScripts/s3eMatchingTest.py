# examples/Python/Basic/file_io.py
import os, torch, json, argparse, shutil
import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import open3d as o3d
import copy
from fsregistration.srv import RequestListPotentialSolution3D
import transforms3d.quaternions as quat




# from src.fsregistration.pythonScripts.matchingProfiling.predator.common.misc import print_info


class MinimalClientAsync(Node):

    def __init__(self, node_name):
        super().__init__('client_' + node_name)
        self.cli = self.create_client(RequestListPotentialSolution3D, 'fs3D/registration/all_solutions')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestListPotentialSolution3D.Request()

    def send_request(self, scan1, scan2, N, VoxelSize, use_clahe, r_min, r_max, set_r_manual, level_potential_rotation,
                     level_potential_translation,normalization_factor):
        self.req.size_of_voxel = VoxelSize
        # self.req.potential_for_necessary_peak = 0.1
        self.req.debug = False
        self.req.dimension_size = N
        self.req.sonar_scan_1 = scan1.tolist()
        self.req.sonar_scan_2 = scan2.tolist()
        self.req.timing_computation_duration = False
        self.req.use_clahe = use_clahe
        self.req.r_min = int(r_min)
        self.req.r_max = int(r_max)
        self.req.level_potential_rotation = level_potential_rotation
        self.req.level_potential_translation = level_potential_translation
        self.req.set_normalization = normalization_factor
        self.req.set_r_manual = set_r_manual

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()



def getVoxelIndex(x, y, z, voxelSize, N):
    voxelX = int(x / voxelSize)
    voxelY = int(y / voxelSize)
    voxelZ = int(z / voxelSize)

    return int(voxelZ + N / 2 + (voxelY + N / 2) * N + (voxelX + N / 2) * N * N)

def getdiffVoxelIndex(x, y, z, voxelSizeX,voxelSizeY,voxelSizeZ, N):
    voxelX = int(x / voxelSizeX)
    voxelY = int(y / voxelSizeY)
    voxelZ = int(z / voxelSizeZ)

    return int(voxelZ + N / 2 + (voxelY + N / 2) * N + (voxelX + N / 2) * N * N)


def pointToVoxelWithColor(pointcloud, N, voxelSize, shift):
    # voxelGrid = Float64MultiArray.data
    voxelGrid = np.zeros(N * N * N)
    voxelGridIndex = np.zeros(N * N * N)
    # print(len(pointcloud.points))
    # print(pointcloud.points[0])
    # print(pointcloud.points[0][0])
    # print(pointcloud.points[1][1])
    # print(pointcloud.points[1][2])
    for indexPoint, point in enumerate(pointcloud.points):
        pointShifted = point - shift
        index = getVoxelIndex(pointShifted[0], pointShifted[1], pointShifted[2], voxelSize, N);

        if index >= 0 and index < len(voxelGrid):
            voxelGrid[index] = voxelGrid[index] + np.mean(np.asarray(pointcloud.colors[indexPoint]))
            voxelGridIndex[index] = voxelGridIndex[index] + 1

        # print(point[0], point[1], point[2])
        # print(index)
    for index in range(N * N * N):
        if voxelGridIndex[index] > 0:
            voxelGrid[index] = voxelGrid[index] / voxelGridIndex[index]
    # getVoxelIndex(x, y, z, voxelSize, N);
    return voxelGrid


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

def pointToVoxelXYZDifference(pointcloud, N, voxelSizeX,voxelSizeY,voxelSizeZ, shift):
    voxelGrid = np.zeros(N * N * N)
    for point in pointcloud.points:
        pointShifted = point - shift
        index = getdiffVoxelIndex(pointShifted[0], pointShifted[1], pointShifted[2], voxelSizeX,voxelSizeY,voxelSizeZ, N);
        if index >= 0 and index < len(voxelGrid):
            voxelGrid[index] = 1

    return voxelGrid

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    source_temp += target_temp
    o3d.io.write_point_cloud("matchingProfiling/resultingTransformation.ply", source_temp, format='auto')
    # o3d.visualization.draw_geometries([source_temp, target_temp])


def draw_pcl_no_color_change(source, transformation, indexFile):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    o3d.io.write_point_cloud("matchingProfiling/resultingTransformation" + str(indexFile) + ".ply", source_temp,
                             format='auto')


def draw_registration_result_no_color_change(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    # source_temp.transform(transformation)
    source_temp += target_temp
    o3d.io.write_point_cloud("matchingProfiling/resultingTransformation.ply", source_temp, format='auto')

def csv_to_pointcloud(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, names=['x', 'y', 'z', 'r', 'g', 'b'])

    # Convert the DataFrame to numpy arrays for coordinates and colors
    points = df[['x', 'y', 'z']].values
    colors = df[['r', 'g', 'b']].values / 255.0  # Normalize RGB values to [0, 1]

    # Create an Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd

def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert angles to radians
    # roll = np.radians(roll)
    # pitch = np.radians(pitch)
    # yaw = np.radians(yaw)

    # Rotation matrices around each axis
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    # Combined rotation matrix
    R = R_z @ R_y @ R_x

    return R

if __name__ == "__main__":


    rclpy.init()

    minimal_client = MinimalClientAsync("myRandomName")
    print("Testing IO for meshes ...")
    whichScan1 = 0
    whichScan2 = 0
    # mesh = o3d.io.read_triangle_mesh("../exampleData/dragon_recon/dragon_vrip.ply")
    # inputPCL = csv_to_pointcloud("/home/tim-external/dataFolder/2016-watertight-meshes-v01/data/becks-barrel/pc.csv")
    # mesh = o3d.io.read_triangle_mesh("../exampleData/dragon_recon/dragon_vrip.ply")
    # demo_colored_icp_pcds = o3d.data.DemoColoredICPPointClouds()
    # ply_point_cloud = o3d.data.PLYPointCloud()
    # pcd1 = o3d.io.read_point_cloud(ply_point_cloud.path)
    # pcd2 = o3d.io.read_point_cloud(ply_point_cloud.path)
    for i in range(20):

        pcd1 = o3d.io.read_point_cloud(f"/home/tim-external/dataFolder/pointclouds/PointcloudAlpha_{whichScan1:05d}.ply")
        pcd2 = o3d.io.read_point_cloud(f"/home/tim-external/dataFolder/pointclouds/PointcloudAlpha_{whichScan2:05d}.ply")

        T = np.eye(4)
        T[:3, 3] = [1.1, 0, 0]
        T[0:3,0:3] = euler_to_rotation_matrix(0,0,i*0.2)
        # T[:3, 3] = [voxelSize * 2 * 1.1, voxelSize * 1, voxelSize * 4]
        # T[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))
        pcd2.transform(T)
        print("transformation Matrix GT: ")
        print(T)
        # print("rotation Quat GT: ")
        # print(quat.mat2quat(T[0:3, 0:3]))


        # print(mesh)
        # mesh.compute_vertex_normals()
        # pcd1 = mesh1.sample_points_uniformly(number_of_points=15000)
        # pcd2 = mesh2.sample_points_uniformly(number_of_points=15000)

        # o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)
        mean1, cov1 = pcd1.compute_mean_and_covariance()
        mean2, cov2 = pcd2.compute_mean_and_covariance()
        N = 64
        # maxDistance = max(np.max(pcd1.points - mean1), np.max(pcd2.points - mean2))
        maxDistance = 20
        voxelSize = (2 * maxDistance * 1.5) / N

        maxDistanceXY = 20
        maxDistanceZ = 20
        voxelSizeXY = (2 * maxDistanceXY * 1.5) / N
        voxelSizeZ = (2 * maxDistanceZ * 1.5) / N

        # print("voxelSizeXY: ",voxelSizeXY)

        mean1 = np.asarray([0,0,0])
        # o3d.geometry.PointCloud.compute_point_cloud_distance
        # mean1 = mean1
        # print(mean1)
        mean2 = mean1
        # print(mean2)
        mean1Transform = np.eye(4)
        mean2Transform = np.eye(4)
        mean1Transform[:3, 3] = np.squeeze(np.asarray(-mean1))
        mean2Transform[:3, 3] = np.squeeze(np.asarray(-mean2))



        # print("percentage Overlap: ", compute_overlap_ratio(pcd1, pcd2, T, voxelSize))
        # pcd1Vox = pcd1.voxel_down_sample(voxel_size=voxelSize)
        # pcd2Vox = pcd2.voxel_down_sample(voxel_size=voxelSize)

        # voxelArray1 = pointToVoxelWithColor(pcd1, N, voxelSize, mean1 + [0, 0, 0]).astype(np.float64)
        # voxelArray2 = pointToVoxelWithColor(pcd2, N, voxelSize, mean2).astype(np.float64)
        # voxelArray1 = pointToVoxel(pcd1, N, voxelSize, mean1 + [0 ,0, 0]).astype(np.float64)
        # voxelArray2 = pointToVoxel(pcd2, N, voxelSize, mean2).astype(np.float64)
        voxelArray1 = pointToVoxelXYZDifference(pcd1, N, voxelSizeXY,voxelSizeXY,voxelSizeZ, mean1).astype(np.float64)
        voxelArray2 = pointToVoxelXYZDifference(pcd2, N, voxelSizeXY,voxelSizeXY,voxelSizeZ, mean2).astype(np.float64)



        # N = 64  # 32 64 128
        use_clahe = True  # True False
        r_min = N / 8  # N / 8 , N / 4
        r_max = N / 2 - N / 8  # N / 2 - N / 8 , N / 2 - N / 4
        set_r_manual = True
        level_potential_rotation = 0.01  # 0.01 , 0.001
        level_potential_translation = 0.1  # 0.1 , 0.01
        normalization_factor = 1  # 0,1
        # ROS2 Node


        response = RequestListPotentialSolution3D.Response()
        response = minimal_client.send_request(voxelArray1, voxelArray2, N, voxelSize, use_clahe, r_min, r_max,
                                               set_r_manual, level_potential_rotation, level_potential_translation,normalization_factor)
        heightFirstPotentialSolution = response.list_potential_solutions[0].transformation_peak_height
        # find highest peak
        # save percentage overlap, angle difference, translation difference,

        highestPeak = 0.0
        indexHighestPeak = 0
        for index, peak in enumerate(response.list_potential_solutions):
            if peak.transformation_peak_height > highestPeak:
                highestPeak = peak.transformation_peak_height
                indexHighestPeak = index

        # print("indexHighestPeak: ", indexHighestPeak)

        peak = response.list_potential_solutions[indexHighestPeak]
        currentQuaternion = [peak.resulting_transformation.orientation.w,
                             peak.resulting_transformation.orientation.x,
                             peak.resulting_transformation.orientation.y,
                             peak.resulting_transformation.orientation.z]
        currentQuaternionInv = quat.qinverse(currentQuaternion)

        np.set_printoptions(suppress=True)
        # print("peak: ", peak.transformation_peak_height / heightFirstPotentialSolution)
        # print("x,y,z: ", peak.resulting_transformation.position)
        print("transformationBefore: ")

        resultingTransformation = np.eye(4)
        resultingTransformation[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(currentQuaternion)
        resultingTransformation[:3, 3] = np.squeeze(np.asarray(
            [peak.resulting_transformation.position.x, peak.resulting_transformation.position.y,
             peak.resulting_transformation.position.z]))

        print(resultingTransformation)
        resultingTransformation[:3, 3] = np.matmul(resultingTransformation[:3, :3],resultingTransformation[:3, 3])
        print("transformationAfter: ")
        print(resultingTransformation)
        # print(currentQuaternion)
        estimatedActualRotation1 = np.matmul(np.linalg.inv(mean2Transform),
                                             np.matmul(resultingTransformation, mean1Transform))

        # print(T)
        # print(estimatedActualRotation1)

        # print(response)
        # draw_registration_result(pcd1, pcd2, T)  # GT
        # draw_registration_result(pcd1, pcd2, estimatedActualRotation1)  # Estimation

        print("test2")
