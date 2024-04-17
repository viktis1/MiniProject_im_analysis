import numpy as np
from scipy.ndimage import convolve1d


def calculates_S(arr, i, j, k, rho, sigma):
    # This function caluclates S for the array in the index arr[i,j,k]
    # rho and sigma are the integration and noise scale respectively.
    # parameters:
    # arr is a numpy arr holding all the voxels.
    # i,j,k are the 1st, 2nd, 3rd coordinate in the array for which the S matrix is being calculated.
    # rho is "integration scale", ie. blurring. It is a variance, NOT a standard devitation.
    # sigma is "noise scale", ie. parameter for the derivative of gauss. It is a variance, NOT a standard devitation.

    cutoff = int(3*np.sqrt(rho))
    x = np.arange(-cutoff, cutoff+1)
    gauss_weights = np.exp(-x**2/(rho*2))
    laplace_weights = -x*np.exp(-x**2/(sigma*2))


    Vx = convolve1d(arr[-cutoff:cutoff+1, :, :], laplace_weights)[cutoff]
    Vy = convolve1d(arr[-cutoff:cutoff+1, :, :], laplace_weights)[cutoff]
    Vz = convolve1d(arr[:, -cutoff:cutoff+1, :], laplace_weights)[cutoff]


    