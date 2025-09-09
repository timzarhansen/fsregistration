#!/usr/bin/python3
import os, json, argparse, shutil

import cv2
# ros include stuff
import rclpy
from rclpy.node import Node
import numpy as np
from fsregistration.srv import RequestListPotentialSolution2D

# from std_msgs.msg import Float64MultiArray
import open3d as o3d
# import copy
import transforms3d.quaternions as quat
import gc


class MinimalClientAsync(Node):

    def __init__(self, node_name):
        super().__init__('client_' + node_name)
        self.cli = self.create_client(RequestListPotentialSolution2D, 'fs2d/registration/all_solutions')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = RequestListPotentialSolution2D.Request()

    def send_request(self, scan1, scan2, N, PixelSize, use_clahe):
        self.req.size_of_pixel = PixelSize
        self.req.potential_for_necessary_peak = 0.5
        self.req.size_image = N
        self.req.sonar_scan_1 = scan1.tolist()
        self.req.sonar_scan_2 = scan2.tolist()
        self.req.multiple_radii = True
        self.req.use_clahe = use_clahe
        self.req.use_hamming = True

        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()


# used to plot everything. now we save to file to look at it from the outside




def getVoxelIndex(x, y, voxelSize, N):
    voxelX = int(x / voxelSize)
    voxelY = int(y / voxelSize)

    return int( voxelY + N / 2 + (voxelX + N / 2) * N)


def load_and_normalize(image_path):
    """Loads grayscale PNG, converts to float array (0-1), returns dimensions."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not read image.")
    height, width = img.shape
    normalized_array = img.astype(np.float32).reshape(-1)  / 255.0
    return normalized_array, (height, width)


# Config:
# 32 1 4 28 0.001 0.001 1
if __name__ == '__main__':
    # load configs
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help='Path to the config file.')
    parser.add_argument('use_clahe', type=int)
    parser.add_argument('potential_for_necessary_peak', type=float, help='Path to the config file.')
    parser.add_argument('size_of_pixel', type=int, help='Path to the config file.')



    args = parser.parse_args()
    N = args.N  # 32 64 128
    use_clahe = bool(args.use_clahe)  # True False
    size_of_pixel = args.size_of_pixel
    potential_for_necessary_peak = args.potential_for_necessary_peak  # 0.01 , 0.001




    # ROS2 Node
    rclpy.init()

    minimal_client = MinimalClientAsync(
        str(N) + '_' + str(int(use_clahe)) + '_' + str(
            int(size_of_pixel * 1000)) + '_' + str(int(potential_for_necessary_peak * 1000)))

    # Load the two Images



    voxelArray1,shapeImage1 = load_and_normalize("../exampleFiles/1547131046353776_radar.png")
    voxelArray2,shapeImage2 = load_and_normalize("../exampleFiles/1547131046606586_radar.png")
    # voxelArray1,shapeImage1 = load_and_normalize("../exampleFiles/1547131049347224_radar.png")
    # voxelArray2,shapeImage2 = load_and_normalize("../exampleFiles/1547131052606390_radar.png")

    response = RequestListPotentialSolution2D.Response()
    response = minimal_client.send_request(voxelArray1, voxelArray2, N, 1.0, use_clahe)
    heightFirstPotentialSolution = response.list_potential_solutions[0].transformation_peak_height

    highestPeak = 0.0
    indexHighestPeak = 0

    for index, peak in enumerate(response.list_potential_solutions):
        # find peak
        if peak.transformation_peak_height > highestPeak:
            highestPeak = peak.transformation_peak_height
            indexHighestPeak = index

    peak = response.list_potential_solutions[indexHighestPeak]
    currentQuaternion = [peak.resulting_transformation.orientation.w,
                         peak.resulting_transformation.orientation.x,
                         peak.resulting_transformation.orientation.y,
                         peak.resulting_transformation.orientation.z]
    # currentQuaternionInv = quat.qinverse(currentQuaternion)

    np.set_printoptions(suppress=True)
    # print("peak: ", peak.transformation_peak_height / heightFirstPotentialSolution)
    # print("x,y,z: ", peak.resulting_transformation.position)
    # print("rotation1: ")

    resultingTransformation = np.eye(4)
    resultingTransformation[:3, :3] = o3d.geometry.get_rotation_matrix_from_quaternion(currentQuaternion)
    resultingTransformation[:3, 3] = np.squeeze(np.asarray(
        [peak.resulting_transformation.position.x, peak.resulting_transformation.position.y,
         peak.resulting_transformation.position.z]))
    print(resultingTransformation)
    # print(currentQuaternion)
    # estimatedActualRotation1 = np.matmul(np.linalg.inv(mean2Transform),
    #                                      np.matmul(resultingTransformation, mean1Transform))
    # print("Estimated Transformation: ")
    # print(estimatedActualRotation1)
    # print("GT Transformation: ")
    # print(T)
    # print("percentage Overlap: ", compute_overlap_ratio(pcd1, pcd2, T, voxelSize))

    # for line in np.matrix(estimatedActualRotation1):
    #     np.savetxt(f, line, fmt='%.10f')

    # print(response)
    # draw_registration_result(pcd1Vox, pcd2Vox, T)  # GT
    # draw_registration_result(pcd1Vox, pcd2Vox, estimatedActualRotation1)#Estimation

        # print("test2")
