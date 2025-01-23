# import kiss_icp.registration
# from kiss_icp.config import KISSConfig
# from kiss_icp.config.config import RegistrationConfig
# from kiss_icp.kiss_icp import KissICP
# from pydantic import BaseModel
#
# assert KissICP is not None
#
# from kiss_icp.registration import get_registration
# import kiss_icp.config
# from kiss_icp.voxelization import voxel_down_sample
# from kiss_icp.threshold import get_threshold_estimator
# from kiss_icp.mapping import get_voxel_hash_map, VoxelHashMap
# from kiss_icp.pybind import kiss_icp_pybind
import numpy as np
import open3d as o3d

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

# def voxelize(iframe, voxelSize):
#     frame_downsample = voxel_down_sample(iframe, voxelSize * 0.5)
#     source = voxel_down_sample(frame_downsample, voxelSize * 1.5)
#     return source, frame_downsample


def register_frame(registration, frame1, frame2, initial_guess, sigma, voxelSize, max_distance, max_points_per_voxel):
    # Apply motion compensation
    # frame = self.compensator.deskew_scan(frame, timestamps, self.last_delta)

    # Preprocess the input cloud
    # frame = self.preprocess(frame)

    # Voxelize
    source1, frame_downsample1 = voxelize(frame1, voxelSize)
    source2, frame_downsample2 = voxelize(frame2, voxelSize)
    local_map = VoxelHashMap(voxelSize, max_distance, max_points_per_voxel)
    local_map.update(frame_downsample2, np.eye(4))

    # Get adaptive_threshold
    # sigma = self.adaptive_threshold.get_threshold()

    # Compute initial_guess for ICP
    # initial_guess = self.last_pose @ self.last_delta

    # Run ICP
    new_pose = registration.align_points_to_map(
        points=source1,
        voxel_map=local_map,
        initial_guess=initial_guess,
        max_correspondance_distance=3 * sigma,
        kernel=sigma / 3,
    )

    # Compute the difference between the prediction and the actual estimate
    # model_deviation = np.linalg.inv(initial_guess) @ new_pose

    # Update step: threshold, local map, delta, and the last pose
    # self.adaptive_threshold.update_model_deviation(model_deviation)
    # self.local_map.update(frame_downsample, new_pose)
    # self.last_delta = np.linalg.inv(self.last_pose) @ new_pose
    # self.last_pose = new_pose

    # Return the (deskew) input raw scan (frame) and the points used for registration (source)
    return new_pose


whichScan1 = 0
whichScan2 = 0

pcd1 = o3d.io.read_point_cloud(f"/home/tim-external/dataFolder/pointclouds/PointcloudAlpha_{whichScan1:05d}.ply")
pcd2 = o3d.io.read_point_cloud(f"/home/tim-external/dataFolder/pointclouds/PointcloudAlpha_{whichScan2:05d}.ply")

# registration = kiss_icp.registration.Registration(500, 0.0001, 1)
voxelSize = 0.5


np.set_printoptions(suppress=True)


T = np.eye(4)
T[:3, 3] = [0.2, 0, 0]
T[0:3,0:3] = euler_to_rotation_matrix(0,0,0.4)
# T[:3, 3] = [voxelSize * 2 * 1.1, voxelSize * 1, voxelSize * 4]
# T[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))
pcd2.transform(T)
print("transformation Matrix GT: ")
print(T)


print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    pcd1, pcd2, voxelSize*2*3, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)
# draw_registration_result(source, target, reg_p2p.transformation)

print("result")


# registration.align_points_to_map()
