import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import tifffile

# Load the image
image = tifffile.imread('multicube.tif').astype(float)/255

test_image = np.empty_like(image)
colors = np.empty(image.shape, dtype=object)

test_image[:,:,:] = False
test_image[image >= 0.7] = True
print("Test A")
print(np.unique(test_image))

colors[test_image.astype(bool)] = "black"
print("Test B")

# Create a figure with 3 subplots
ax = plt.figure().add_subplot(projection='3d')
print("Test C")
ax.scatter(*np.where(test_image.astype(bool)), c=colors[test_image.astype(bool)], s=1, alpha=0.1)
print("Test D")
plt.show()

