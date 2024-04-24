import numpy as np
import matplotlib.pyplot as plt
import tifffile
from Helper_functions import calculates_S, construct_S, eigendecomposition, anisotropy
from tqdm import tqdm

# Load the image
im = tifffile.imread('multicube.tif').astype(float)/255
S_ish = calculates_S(im, 7, 5)


# Initialize arrays to hold dominant direction and shape measures
best_direction = np.zeros((im.shape[0], im.shape[1], im.shape[2], 3))
shape_measure = np.zeros((im.shape[0], im.shape[1], im.shape[2], 3))

for x in tqdm(range(S_ish.shape[0])):
    for y in range(S_ish.shape[1]):
        for z in range(S_ish.shape[2]):
            # Find S-matrix
            S = construct_S(S_ish[x,y,z,:])
            # Find eigenvalues and -vectors
            lam, vec = eigendecomposition(S)
            c = anisotropy(lam)

            best_direction[x,y,z,:] = vec # Load best direction
            shape_measure[x,y,z,:] = c # Load shape measures


np.save('best_direction.npy', best_direction)
np.save('shape_measure.npy', shape_measure)
