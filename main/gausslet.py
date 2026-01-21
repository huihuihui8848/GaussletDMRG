import numpy as np
import os
from tools import read_coef, make_centers_and_scale

def gaussian(x, mu=0.0, sigma=1.0):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = float(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gausslet_primary(x, b):
    """
    The primitive gausslet combination:
        phi(x) = sum_j b_j [ G(x*3 - j) + G(x*3 + j) ] (j>0)
               + b_0 G(x*3)
    """
    x = np.asarray(x, dtype=float)
    x = np.atleast_1d(x)
    x3 = 3.0 * x
    b = np.asarray(b, dtype=float)
    j = np.arange(len(b), dtype=float)

    res = b[0] * gaussian(x3, j[0], sigma=1.0)
    if j.size > 1:
        Gpos = gaussian(x3[:, None],  j[1:], sigma=1.0)
        Gneg = gaussian(x3[:, None], -j[1:], sigma=1.0)
        res = res + Gpos @ b[1:] + Gneg @ b[1:]
    return np.squeeze(res)

def gausslet_scaled(x, b, center=0.0, s=1.0):
    """
    Scaled + Translated basis function:
        phi(x; center, s) = s^{-1/2} * gausslet_primary((x-center)/s)
    """
    x = np.asarray(x, dtype=float)
    y = (1.0 / np.sqrt(s)) * gausslet_primary((x - center) / s, b)
    return np.squeeze(y)

if __name__ == '__main__':
    L = 10
    N = 5
    x = np.arange(start=-L, stop=L, step=0.01)

    b = read_coef('coefficient/G4.csv')

    centers, scale = make_centers_and_scale(L, N)

    import matplotlib.pyplot as plt
    for center in centers:
        plt.plot(x, gausslet_scaled(x, b, center, scale))
    plt.show()