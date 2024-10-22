import os, torch, json, argparse, shutil
from predator.datasets.dataloader import get_dataloader, get_datasets
from easydict import EasyDict as edict
from predator.lib.utils import setup_seed, load_config
# ros include stuff
import rclpy
from rclpy.node import Node
import numpy as np
from fsregistration.srv import _request_list_potential_solution3_d
import open3d as o3d
import copy
class MinimalService(Node):

    def __init__(self):
        super().__init__('minimal_service')
        self.srv = self.create_service(AddTwoInts, 'add_two_ints', self.add_two_ints_callback)

    def add_two_ints_callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))

        return response

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])



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
    parser.add_argument('config', type=str, help= 'Path to the config file.')
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
                                                              shuffle=True,
                                                              num_workers=config.num_workers,
                                                              )
    config.val_loader, _ = get_dataloader(dataset=val_set,
                                           batch_size=config.batch_size,
                                          shuffle=False,
                                          num_workers=0,
                                          neighborhood_limits=neighborhood_limits
                                          )
    config.test_loader, _ = get_dataloader(dataset=benchmark_set,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=0,
                                           neighborhood_limits=neighborhood_limits)




    for _ in range(10):
        inputs = next(iter(config.train_loader))

        # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
        pcd1 = o3d.geometry.PointCloud()
        pcd2 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(inputs["src_pcd_raw"])
        pcd2.points = o3d.utility.Vector3dVector(inputs["tgt_pcd_raw"])
        # draw_registration_result(pcd1,pcd2,np.identity(4))

        T = np.eye(4)
        T[:3, :3] = inputs["rot"]
        T[:3, 3] = np.squeeze(np.asarray(inputs["trans"]))
        print(T)
        draw_registration_result(pcd1,pcd2,T)

        print("test")








