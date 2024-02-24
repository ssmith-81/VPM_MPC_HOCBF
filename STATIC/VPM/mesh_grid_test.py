import numpy as np

# Could plot the stored data in SITL (not hardware) if desired:
import matplotlib.pyplot as plt
from matplotlib import path

# Grid parameters
nGridX = 20  # X-grid for streamlines and contours
nGridY = 20  # Y-grid for streamlines and contours

# Define the lidar parameters
lidar_range = 3.5  # Lidar range in meters
lidar_resolution = 0.05  # Lidar resolution in meters
# Define random lidar arrays for combination
x_l = np.array([2.5,5.5,10.5])
y_l = np.array([3.5,6,13])

# Define the grid extents
xVals = [-1, 21]  # X-grid extents [min, max]
yVals = [-1, 21]  # Y-grid extents [min, max]

# Generate the coarse grid points
XX, YY = np.meshgrid(np.linspace(xVals[0], xVals[1], nGridX),
                                          np.linspace(yVals[0], yVals[1], nGridY))
                                          
# Concatenate XX with xa and xa_orig
XX_new = np.concatenate((XX[0, :], x_l))

# Concatenate YY with ya and ya_orig
YY_new = np.concatenate((YY[:, 0], y_l))

# Sort XX_new and YY_new individually
XX_sorted = np.sort(XX_new)
YY_sorted = np.sort(YY_new)

 # Create mesh grids from the sorted XX_sorted and YY_sorted
XX_mesh, YY_mesh = np.meshgrid(XX_sorted, YY_sorted)


print(XX_sorted)
print(YY_sorted)



plt.figure(5)
plt.scatter(XX_mesh,YY_mesh,color = 'blue', label='data-points')
plt.xlabel('x-data')
plt.ylabel('y-data')
plt.grid(True)
plt.legend()
plt.show()
