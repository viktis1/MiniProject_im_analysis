import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib.colors import 
from time import perf_counter

# Measre start time
start = perf_counter()


shape_measure = np.load('shape_measure.npy')
best_direction = np.load('best_direction.npy')

place_holder = np.ones((shape_measure.shape[0], shape_measure.shape[1], shape_measure.shape[2]), dtype=bool)
mask = np.argmax(shape_measure, axis=3)


# COLORMAP PLAYTIME
cmap = cmap = cm.get_cmap('viridis', 3)  

# Plot the results
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(*np.where(place_holder), c=mask, s=1, alpha=0.01, cmap=cmap)
# To add a colorbar, we dont want the alpha-value to affect the colorbar, so we will plot points not in image
p = ax.scatter([900,900,900], [900,900,900], [900,900,900], c=[0,1,2], cmap=cmap)
cbar = fig.colorbar(p, ax=ax, ticks=[0.33, 1, 1.66])
cbar.ax.set_yticklabels(['Linearity', 'Planarity', 'Sphericity'])
# Now we have to cut out the parts we dont want to see...
ax.set_xlim(-10,160)
ax.set_ylim(-10, 160)
ax.set_zlim(-10, 160)
# Save figure. This takes time (i think because it has so many points?). This part is roughly 70 times as time-consuming as the rest.
plt.savefig('shape_measures.png')
print(perf_counter() - start)