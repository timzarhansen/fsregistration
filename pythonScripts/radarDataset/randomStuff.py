import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pyboreas import BoreasDataset
from pyboreas.utils.utils import get_inverse_tf
import rclpy
from rclpy.node import Node
from fsregistration.srv import RequestListPotentialSolution2D

np.set_printoptions(precision=5, suppress=True)

def vis_radar(
    rad,
    figsize=(10, 10),
    dpi=100,
    cart_resolution=0.2384,
    cart_pixel_width=640,
    cmap="gray",
    show=True,
    save=None,
):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot()
    ax.imshow(rad, cmap=cmap)
    ax.set_axis_off()
    if show:
        plt.show()
    if save is not None:
        plt.savefig(save, bbox_inches="tight")
    return ax


root = '/home/tim-external/dataFolder/radar_boreas/'
split = None
# AWS: Note: Free Tier SageMaker instances don't have enough storage (25 GB) for 1 sequence (100 GB)
# root = '/home/ec2-user/SageMaker/boreas/'
# split = [['boreas-2021-09-02-11-42', 163059759e6, 163059760e6-1]]

# With verbose=True, the following will print information about each sequence
bd = BoreasDataset(root, split=split, verbose=True)
# Grab the first sequence
seq = bd.sequences[1]

seq.calib.print_calibration()


# matplotlib inline
# Let's visualize a lidar frame:
cart_pixel_width = 512
cart_resolution = 0.75
lid = seq.get_lidar(0)
lid.passthrough([-40, 40, -40, 40, -10, 30])
lid.visualize(figsize=(10, 10), color='intensity', vmin=5, vmax=40)
cam = seq.get_camera(0)
cam.visualize(figsize=(10, 10))
rad1 = seq.get_radar(0)
rad2 = seq.get_radar(10)
cart1 = rad1.polar_to_cart(
    cart_resolution=cart_resolution,
    cart_pixel_width=cart_pixel_width,
    in_place=False,
)
cart2 = rad2.polar_to_cart(
    cart_resolution=cart_resolution,
    cart_pixel_width=cart_pixel_width,
    in_place=False,
)

# vis_radar(rad=cart1,cart_resolution=cart_resolution,cart_pixel_width=cart_pixel_width)
# vis_radar(rad=cart2,cart_resolution=cart_resolution,cart_pixel_width=cart_pixel_width)
print(rad1.pose)
print(rad2.pose)
# lid = seq.get_lidar(0)






