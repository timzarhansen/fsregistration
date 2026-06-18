import numpy as np


def radar_to_point_cloud(image, threshold, pixel_size, color):
    # Get image dimensions (assuming square image)
    N = image.shape[0]

    # Compute grid coordinates centered at the image center
    center_offset = N / 2 + 0.5
    x = ((np.arange(1, N + 1)) - center_offset) * pixel_size
    y = ((np.arange(1, N + 1)) - center_offset) * pixel_size

    # Create coordinate grids
    X, Y = np.meshgrid(x, y)

    # Find indices where the image exceeds the threshold
    ind = image > threshold

    # Extract valid points and their intensities
    points = np.column_stack([X[ind], Y[ind], np.zeros(np.sum(ind))])
    intensity = image[ind]

    # Create point cloud with intensity as an attribute
    return_pc = np.hstack([points, np.tile(color, (len(points), 1))])

    return return_pc
