import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random 3D data
num_points = 100
x_data = np.random.randn(num_points, 1)
y_data = np.random.randn(num_points, 1)
z_data = np.random.randn(num_points, 1)

# Define parameters for the 3D Gaussian surface
mu_x = np.mean(x_data)  # Mean of x
mu_y = np.mean(y_data)  # Mean of y
mu_z = np.mean(z_data)  # Mean of z
sigma_x = np.std(x_data)  # Standard deviation of x
sigma_y = np.std(y_data)  # Standard deviation of y
sigma_z = np.std(z_data)  # Standard deviation of z

# Create a grid for the 3D Gaussian surface
x, y = np.meshgrid(np.arange(-4, 4.1, 0.1), np.arange(-4, 4.1, 0.1))
z_grid = np.exp(-((x - mu_x) ** 2 / (2 * sigma_x ** 2) + (y - mu_y) ** 2 / (2 * sigma_y ** 2)))

# Plot the random data and the 3D Gaussian surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.hold(True)

# Scatter plot of the random data
ax.scatter(x_data, y_data, z_data, c='gray', alpha=0.5)

# Surface plot of the 3D Gaussian curve
ax.plot_surface(x, y, z_grid, alpha=0.5, edgecolor='none')

# Add labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Random 3D Data with 3D Gaussian Surface')

ax.hold(False)
