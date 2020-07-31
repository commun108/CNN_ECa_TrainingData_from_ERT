import numpy as np


def sensitivityHCP(depth, coiloffset):
    """
    Computes the cumulative response function for
    a horizontal coil orientation (HCP).
    McNeill (1980)

    Input:
    depth - double, depth below sensor
    coiloffset - double, coil separation

    Returns:
    phi_hcp - double, cum. response for offset and depth
    """
    return 1 / (np.sqrt(4 * (depth / coiloffset)**2 + 1))


def sensitivityVCP(depth, coiloffset):
    """
    Computes the cumulative response function for
    a vertical coil orientation (HCP).
    McNeill (1980)

    Input:
    depth - double, depth below sensor
    coiloffset - double, coil separation

    Returns:
    phi_vcp - double, cum. response for offset and depth
    """
    return np.sqrt(4 * (depth / coiloffset)**2 + 1) - 2 * (depth / coiloffset)


def computeTempSum(sensitivity_function, layer_depths, layer_conds, coiloffset):
    """
    Helper function to compute sum of cumulative responses for different depths
    and conductivity values.
    """

    Re = sensitivity_function(layer_depths[:-1], coiloffset)
    Ru = sensitivity_function(layer_depths[1:], coiloffset)

    return np.sum(layer_conds * (Re - Ru))


def imagMagFieldResponse(layer_depths, layer_conds, coiloffset, angular_freq, magn_perm=4 * np.pi * 1e-7):
    """
    Computes the magnetic field response for the n layer with layer_depths and layer_conds for a given
    coiloffset and angular frequency.

    Input:
    layer_depths - array, depths of the layer interfaces, contains 0 as surface
    layer_conds - array, conductivity values of the layers in S/m
    coiloffset - double, coil separation in m
    angular_freq - double, angular frequency of instrument
    magn_perm - double, magnetic permeability of free space

    Returns:
    imagH_vcp - magnetic field response for vertical coil orientation
    imagH_hcp - magnetic field response for horizontal coil orientation
    """

    imagH_hcp = (angular_freq * magn_perm * coiloffset**2) / 4 * \
        computeTempSum(sensitivityHCP, layer_depths, layer_conds, coiloffset)
    imagH_vcp = (angular_freq * magn_perm * coiloffset**2) / 4 * \
        computeTempSum(sensitivityVCP, layer_depths, layer_conds, coiloffset)

    return imagH_hcp, imagH_vcp


def computeForwardResponse(layer_depths, layer_conds, coiloffset, angular_freq, magn_perm=4 * np.pi * 1e-7):
    """
    Using the low induction number approximation compute the apparent conductivity
    for a given layer model and coil separation.
    """

    imagH_hcp, imagH_vcp = imagMagFieldResponse(
        layer_depths, layer_conds, coiloffset, angular_freq)
    sigma_hcp = (4 * imagH_hcp) / (angular_freq * magn_perm * coiloffset**2)
    sigma_vcp = (4 * imagH_vcp) / (angular_freq * magn_perm * coiloffset**2)

    return sigma_hcp, sigma_vcp


layer_depths = np.array(
    [0, 0.2, 0.43, 0.7, 1.01, 1.37, 1.79, 2.27, 2.83, 3.47, 4.22, 5.08, 6.])
layer_conds = np.array([0.013, 0.013, 0.013, 0.01319889, 0.01363138,
                        0.01468311, 0.01596775, 0.01704566, 0.01893862, 0.0206872,
                        0.02132148, 0.022])

coiloffset = [1.48, 2.82, 4.49]

a, b = computeForwardResponse(
    layer_depths, layer_conds, coiloffset[2], 2 * np.pi * 10000)
