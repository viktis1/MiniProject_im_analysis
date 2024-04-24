import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
import tifffile

# Load the image
image = tifffile.imread('multicube.tif').astype(float)/255

test_image = np.ones(image.shape, dtype=bool)

x = np.where(test_image)[0]
y = np.where(test_image)[1]
z = np.where(test_image)[2]

# Create a 3d plot
ax = plt.figure().add_subplot(projection='3d')
print("Plotting...  ", end="")
ax.scatter(x,y,z, c=image.ravel(), s=1, alpha=1, cmap='gray')
print("Done !")
plt.savefig('Dataset_easy.png')
# plt.show()

