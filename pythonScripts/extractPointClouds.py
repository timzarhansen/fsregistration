import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import open3d as o3d
import numpy as np
import os
import signal
import sensor_msgs_py.point_cloud2 as pc2



def convert_point_cloud_msg_to_numpy( data: PointCloud2):
    if data is not None:
        print('test message 1')
        # Parse the PointCloud2 message
        gen = pc2.read_points(data, skip_nans=True)
        pointsPCL = np.zeros((len(gen),3))
        print(gen.shape)
        for x in range(len(gen)):
            pointsPCL[x,0] = gen[x][0]
            pointsPCL[x, 1] = gen[x][1]
            pointsPCL[x, 2] = gen[x][2]
            # pointsPCL[x, 3] = gen[x][3]
        # testArray = np.fromiter(gen, float)
        # tmpPC = o3d.utility.Vector3dVector(pointsPCL)
        # print(gen.reshape(-1,3).shape)
        # Convert the point cloud to a numpy array
        # points_numpy = np.array(list(data), dtype=np.float32)

        return pointsPCL


class PointCloudSaver(Node):
    def __init__(self):
        super().__init__('pointcloud_saver')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/Alpha/velodyne_points',
            self.listener_callback,
            10)
        self.count = 0
        os.makedirs('/home/tim-external/dataFolder/pointclouds', exist_ok=True)

    def listener_callback(self, msg):
        pointCloudOut = convert_point_cloud_msg_to_numpy(msg)
        # pc_data = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointCloudOut)
        file_name = f'/home/tim-external/dataFolder/pointclouds/PointcloudAlpha_{self.count:05d}.ply'
        o3d.io.write_point_cloud(file_name, pcd)
        self.get_logger().info(f'Saved point cloud to {file_name}')
        self.count += 1

def main(args=None):
    rclpy.init(args=args)
    node = PointCloudSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
