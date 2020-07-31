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


def computePropagationConstant(angular_freq, cond, magn_perm=4 * np.pi * 1e-7, vacuum_perm=8.8541878128 * 1e-12):
    """
    """
    # return np.sqrt(angular_freq * magn_perm * (1j * cond - angular_freq * vacuum_perm)) # mester 2011
    return np.sqrt(angular_freq * magn_perm * 1j * cond)  # hebel et al 2013


def computeGamma(wavenumber, angular_freq, cond, magn_perm=4 * np.pi * 1e-7, vacuum_perm=8.8541878128 * 1e-12):
    """
    """
    propagation_constant = computePropagationConstant(
        angular_freq, cond, magn_perm, vacuum_perm)
    return np.sqrt(wavenumber**2 + propagation_constant**2)


def computeReflectionCoefficient(layer_thickness, layer_conds, wavenumber, angular_freq, magn_perm=4 * np.pi * 1e-7, vacuum_perm=8.8541878128 * 1e-12):
    """
    """

    Rstart = 0
    gamma = computeGamma(wavenumber, angular_freq, layer_conds)
    print(Rstart)
    print((gamma))

    for idx, h in enumerate(layer_thickness):
        # print(idx)
        if idx == len(layer_thickness) - 1:
            break

        # nominator = ((gamma[idx] - gamma[idx + 1]) / (gamma[idx] +
        #                                               gamma[idx + 1])) + Rstart * np.exp(-2 * gamma[idx + 1] * h)
        # denominator = 1 - ((gamma[idx] - gamma[idx + 1]) / (gamma[idx] +
        #                                                     gamma[idx + 1])) + Rstart * np.exp(-2 * gamma[idx + 1] * h)
        nominator = ((gamma[idx] - gamma[idx + 1]) / (gamma[idx] +
                                                      gamma[idx + 1])) + Rstart * np.exp(-2 * gamma[idx + 1] * layer_thickness[idx + 1])
        denominator = 1 - ((gamma[idx] - gamma[idx + 1]) / (gamma[idx] +
                                                            gamma[idx + 1])) * Rstart * np.exp(-2 * gamma[idx + 1] * layer_thickness[idx + 1])

        Rstart = nominator / denominator
        print(Rstart)

    return Rstart


def computeMagFieldResponse(coiloffset, reflection_coefficient, wavenumber):
    """
    .
    """
    from scipy.special import jv
    from scipy.integrate import quad_vec

    def integrand(wavenumber):
        return np.imag(reflection_coefficient) * jv(0, wavenumber * coiloffset) * wavenumber**2

    integrated, error = quad_vec(integrand, 0, np.inf)

    return 1 - coiloffset**3 * integrated


def computeForwardResponse(magfieldresponse, coiloffset, angular_freq, magn_perm=4 * np.pi * 1e-7):
    """
    Using the low induction number approximation compute the apparent conductivity
    for a given layer model and coil separation.
    """

    sigma_hcp = (4 * magfieldresponse) / \
        (angular_freq * magn_perm * coiloffset**2)
    # sigma_vcp = (4 * imagH_vcp) / (angular_freq * magn_perm * coiloffset**2)

    return sigma_hcp


layer_depths = np.array(
    [0, 0.2, 0.43, 0.7, 1.01, 1.37, 1.79, 2.27, 2.83, 3.47, 4.22, 5.08, 6.])
layer_thickness = np.diff(layer_depths)
layer_conds = np.array([0.013, 0.013, 0.013, 0.01319889, 0.01363138,
                        0.01468311, 0.01596775, 0.01704566, 0.01893862, 0.0206872,
                        0.02132148, 0.022])

# coiloffset = [1.48, 2.82, 4.49]
#
# a, b = computeForwardResponse(
#     layer_depths, layer_conds, coiloffset[2], 2 * np.pi * 10000)

angular_freq = 2 * np.pi * 1e4
vacuum_c = 299792458
wavenumber = angular_freq / vacuum_c
# wavenumber is wavelength
wavelength = vacuum_c / 1e4
# wavenumber = wavelength
coiloffset = 1.42

a = computePropagationConstant(angular_freq, layer_conds[0])
b = computeGamma(wavenumber, angular_freq, layer_conds[0])
c = computeReflectionCoefficient(
    layer_thickness, layer_conds, wavenumber, angular_freq)
# d = computeGamma(wavenumber, angular_freq, layer_conds[0])
# e = computeGamma(wavenumber, angular_freq, layer_conds[1])

d = computeMagFieldResponse(coiloffset, c, wavenumber)
e = computeForwardResponse(d, coiloffset, angular_freq)
