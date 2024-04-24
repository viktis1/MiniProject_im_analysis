import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib.colors import 
from time import perf_counter

# Measre start time
start = perf_counter()
# Load data
shape_measure = np.load('shape_measure.npy')
best_direction = np.load('best_direction.npy')

##########
# 1st plot
##########

place_holder = np.ones((shape_measure.shape[0], shape_measure.shape[1], shape_measure.shape[2]), dtype=bool)
mask = shape_measure[:,:,:,0] > 0.5

test_array = np.stack([mask, mask,mask], axis=3).astype(bool)
color_dir = best_direction[test_array].reshape(-1,3)


# Define the colors to be plotted 
max_rgb = np.max(np.abs(color_dir))
cr = (max_rgb + color_dir[:,0]) / (2*max_rgb)
cg = (max_rgb + color_dir[:,1]) / (2*max_rgb)
cb = (max_rgb + color_dir[:,2]) / (2*max_rgb)
# Stack 'em high
colorspace = [(r,b,g) for r,b,g in zip(cr.ravel(), cg.ravel(), cb.ravel())]
place_holder[mask == False] = False


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sp = ax.scatter(*np.where(mask), c=colorspace, s=0.2, alpha=1)
ax.set_xlim(-10,160)
ax.set_ylim(-10, 160)
ax.set_zlim(-10, 160)
# Save figure. This takes time (i think because it has so many points?). This part is roughly 70 times as time-consuming as the rest.
plt.savefig('Linearity.png')
print(f"1st plot took: {perf_counter() - start} sec")



##########
# 2nd plot - Quiver for Linearity
##########
mask = shape_measure[:,:,:,0] > 0.8

x = np.where(mask)[0]
y = np.where(mask)[1]
z = np.where(mask)[2]
u = best_direction[:,:,:,0][mask].ravel()
v = best_direction[:,:,:,1][mask].ravel()
w = best_direction[:,:,:,2][mask].ravel()

# Convert best direction to spherical coordinates
rho = np.sqrt(u**2 + v**2 + w**2)
theta = np.arctan2(v,u)
phi = np.arccos(w/rho)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.quiver(x,y,z, u,v,w, length=1)
plt.savefig('linearity_quiver.png')
print(f"2nd plot took: {perf_counter() - start} sec")

