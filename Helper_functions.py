import numpy as np
from scipy.ndimage import convolve1d

def calculates_S(arr, rho, sigma):
    # This function caluclates S for the array in the index arr[i,j,k]
    # rho and sigma are the integration and noise scale respectively.
    # parameters:
    # arr is a numpy arr holding all the voxels.
    # rho is "integration scale", ie. blurring. It is a variance, NOT a standard devitation.
    # sigma is "noise scale", ie. parameter for the derivative of gauss. It is a variance, NOT a standard devitation.

    cutoff = int(3*np.sqrt(rho))

    x = np.arange(-cutoff, cutoff+1)
    gauss_weights = np.exp(-x**2/(rho*2))
    laplace_weights = -x*np.exp(-x**2/(sigma*2))

    # Compute the gradients. A Gaussian prior is used to smooth the picture before gradient is taken. Due to the separability of Gaussian kernels, this corresponds to convolving with the derivative of the Gaussian filter.
    Vx = convolve1d(arr, laplace_weights, axis=0)
    Vy = convolve1d(arr, laplace_weights, axis=1)
    Vz = convolve1d(arr, laplace_weights, axis=2)

    # Define the 6 volumes
    Vxx = Vx*Vx
    Vyy = Vy*Vy
    Vzz = Vz*Vz
    Vxy = Vx*Vy
    Vxz = Vx*Vz
    Vyz = Vy*Vz

    # Create a helper function that convolves the image in all 3 directions.
    def convolve3d(arr, gauss_weights):
        # Convolve over all dimensions.
        for i in range(3):
            arr = convolve1d(arr, gauss_weights, axis=i)
        return arr

    # Define elements of S
    sxx = convolve3d(Vxx, gauss_weights)
    syy = convolve3d(Vyy, gauss_weights)
    szz = convolve3d(Vzz, gauss_weights)
    sxy = convolve3d(Vxy, gauss_weights)
    sxz = convolve3d(Vxz, gauss_weights)
    syz = convolve3d(Vyz, gauss_weights)

    return np.stack((sxx, syy, szz, sxy, sxz, syz), axis=3)



def construct_S(S_elements):
    # The input to this function is a vector with the elements: [sxx, syy, szz, sxy, sxz, syz].
    assert S_elements.size==6, "Construct_S only accepts 1 vector at a time [sxx, syy, szz, sxy, sxz, syz]"
    
    # Define the elements of S
    sxx = S_elements[0]; syy = S_elements[1]; szz = S_elements[2]
    sxy =S_elements[3]; sxz = S_elements[4]; syz = S_elements[5]
    # Return normalized S
    S = np.array([[sxx, sxy, sxz], [sxy, syy, syz], [sxz, syz, szz]])
    return S/np.trace(S)

def eigendecomposition(S):
    # Find the eigenvalues and the dominant direction (eigenvector corresponding to smallest eigenvalue)
    assert S.shape==(3,3), "eigendecomposition only takes 1 (3x3)-matrix as input"

    lam, vec = np.linalg.eig(S)
    # Find sorted list of eigenvalues
    idx = np.argsort(lam)
    lam = lam[idx]
    vec = vec[idx]
    return lam, vec[:, 0]



def anisotropy(eig_vals):
    assert eig_vals.size==3, "anisotropy only works with 3 eigen-values"
    
    # Define the eigenvalues and then the shape-measures
    l1 = eig_vals[0]; l2 = eig_vals[1]; l3 = eig_vals[2]
    cl = (l2-l1)/l3 # Measure of linearity
    cp = (l3-l2)/l3 # Measure of planarity
    cs = l1/l3 # Measure of sphericity
    
    return np.array([cl, cp, cs])



    