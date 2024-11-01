# examples/Python/Basic/file_io.py
import os, torch, json, argparse, shutil
import rclpy
from rclpy.node import Node
import numpy as np
import open3d as o3d
import copy
from fsregistration.srv import RequestListPotentialSolution3D
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
        self.req.debug = True
        self.req.dimension_size = N
        self.req.sonar_scan_1 = scan1.tolist()
        self.req.sonar_scan_2 = scan2.tolist()
        self.req.timing_computation_duration = False
        self.req.use_clahe = True

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

def getVoxelIndex(x, y, z, voxelSize, N):
    voxelX = int(x / voxelSize)
    voxelY = int(y / voxelSize)
    voxelZ = int(z / voxelSize)

    return int(voxelZ + N / 2 + (voxelY + N / 2) * N + (voxelX + N / 2) * N * N)


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

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



if __name__ == "__main__":



    print("Testing IO for meshes ...")
    mesh = o3d.io.read_triangle_mesh("../exampleData/dragon_recon/dragon_vrip.ply")
    print(mesh)
    mesh.compute_vertex_normals()
    pcd1 = mesh.sample_points_uniformly(number_of_points=15000)
    pcd2 = mesh.sample_points_uniformly(number_of_points=15000)



    # o3d.io.write_triangle_mesh("copy_of_knot.ply", mesh)
    mean1, cov1 = pcd1.compute_mean_and_covariance()
    mean2, cov2 = pcd2.compute_mean_and_covariance()
    N = 64
    maxDistance = max(np.max(pcd1.points-mean1),np.max(pcd2.points-mean2))
    voxelSize = (2*maxDistance*1.5)/N

    # o3d.geometry.PointCloud.compute_point_cloud_distance
    # mean1 = mean1
    print(mean1)
    print(mean2)
    mean1Transform = np.eye(4)
    mean2Transform = np.eye(4)
    mean1Transform[:3, 3] = np.squeeze(np.asarray(-mean1))
    mean2Transform[:3, 3] = np.squeeze(np.asarray(-mean2))

    # draw_registration_result(pcd1,pcd2,np.identity(4))

    T = np.eye(4)
    T[:3, 3] = [voxelSize*2*1.1,voxelSize*1,voxelSize*4]
    # T[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))
    pcd2.transform(T)
    print("rotation Matrix GT: ")
    print(T)
    print("rotation Quat GT: ")
    print(quat.mat2quat(T[0:3, 0:3]))

    # print("percentage Overlap: ", compute_overlap_ratio(pcd1, pcd2, T, voxelSize))
    pcd1Vox = pcd1.voxel_down_sample(voxel_size=voxelSize)
    pcd2Vox = pcd2.voxel_down_sample(voxel_size=voxelSize)

    voxelArray1 = pointToVoxel(pcd1, N, voxelSize, mean1 + [0 ,0, 0]).astype(np.float64)
    voxelArray2 = pointToVoxel(pcd2, N, voxelSize, mean2).astype(np.float64)



    # ROS2 Node
    rclpy.init()

    minimal_client = MinimalClientAsync()



    response = RequestListPotentialSolution3D.Response()
    response = minimal_client.send_request(voxelArray1, voxelArray2, N, voxelSize)
    heightFirstPotentialSolution = response.list_potential_solutions[0].transformation_peak_height
    # find highest peak
    # save percentage overlap, angle difference, translation difference,



    highestPeak = 0.0
    indexHighestPeak = 0
    for index,peak in enumerate(response.list_potential_solutions):
        if peak.transformation_peak_height> highestPeak:
            highestPeak = peak.transformation_peak_height
            indexHighestPeak = index

    print("indexHighestPeak: ", indexHighestPeak)


    peak = response.list_potential_solutions[indexHighestPeak]
    currentQuaternion = [peak.resulting_transformation.orientation.w,
                         peak.resulting_transformation.orientation.x,
                         peak.resulting_transformation.orientation.y,
                         peak.resulting_transformation.orientation.z]
    currentQuaternionInv = quat.qinverse(currentQuaternion)

    np.set_printoptions(suppress=True)
    print("peak: ", peak.transformation_peak_height / heightFirstPotentialSolution)
    print("x,y,z: ", peak.resulting_transformation.position)
    print("rotation1: ")

    resultingTransformation = np.eye(4)
    resultingTransformation[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(currentQuaternion)
    resultingTransformation[:3, 3] = np.squeeze(np.asarray(
        [peak.resulting_transformation.position.x, peak.resulting_transformation.position.y,
         peak.resulting_transformation.position.z]))
    print(resultingTransformation[:3, :3])
    print(currentQuaternion)
    estimatedActualRotation1 = np.matmul(np.linalg.inv(mean2Transform), np.matmul(resultingTransformation,mean1Transform))
    print(estimatedActualRotation1)
    print(T)
    # print(response)
    draw_registration_result(pcd1, pcd2, T)#GT
    draw_registration_result(pcd1, pcd2, estimatedActualRotation1)#Estimation

    print("test2")



