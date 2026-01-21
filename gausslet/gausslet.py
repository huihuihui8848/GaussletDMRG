import numpy as np
from matplotlib import pyplot as plt

try:
    from .metrics import overlap_matrix  # type: ignore
except ImportError:  # fallback when running as a script
    from metrics import overlap_matrix  # type: ignore


def gaussian(x, mu=0.0, sigma=1.0):
    """Unnormalized Gaussian: exp(-0.5 * ((x-mu)/sigma)^2)."""
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = float(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def gausslet_primary(x, b):
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
    x = np.asarray(x, dtype=float)
    y = (1.0 / np.sqrt(s)) * gausslet_primary((x - center) / s, b)
    return np.squeeze(y)


def build_basis(centers, b, s: float = 1.0):
    """
    Return a list of callables f_i(x) = gausslet_scaled(x, b, center=centers[i], s=s).
    Typically choose centers as centers = s * np.arange(k0, k1) for near-orthogonality.
    """
    centers = list(centers)
    return [lambda x, c=c: gausslet_scaled(x, b, center=c, s=s) for c in centers]


def _default_L(centers, b, s: float) -> float:
    """Heuristic half-domain size for numerical quadrature."""
    centers = np.asarray(centers, dtype=float)
    b = np.asarray(b, dtype=float)
    J = len(b) - 1
    sigma_x = s / 3.0
    far_internal = (J / 3.0) * s          # farthest internal sub-Gaussian offset from its gausslet center
    tail = 6.0 * sigma_x                  # ~6 sigma covers essentially all mass
    return float(np.max(np.abs(centers)) + far_internal + tail)


# ---------------------------- demo / usage ----------------------------
if __name__ == "__main__":
    coe = "coefficient/G4.csv"

    # Robustly load coefficients: accept 1-col (b) or 2-col (j,b)
    arr = np.loadtxt(coe, delimiter=",")
    if arr.ndim == 1:   # single column
        b = arr.astype(float)
    else:
        # assume first column is j (0..M-1), second column is b
        # sort by j just in case
        order = np.argsort(arr[:, 0])
        j_in = arr[order, 0].astype(float)
        b = arr[order, 1].astype(float)
        # optional sanity: j_in should be 0..M-1 (within tol); otherwise we still proceed
        # but you can assert if you require exact integer grid:
        # assert np.allclose(j_in, np.arange(len(b), dtype=float))

    # choose scale s and centers (recommended step = s)
    s = 1.0
    centers = s * np.arange(-4, 5, 1)   # [-4s, ..., 4s]

    # plot one scaled gausslet at c=0
    xx = np.linspace(-4*s, 4*s, 800)
    yy = gausslet_scaled(xx, b, center=0.0, s=s)

    plt.plot(xx, yy, lw=2)
    plt.xlabel("x")
    plt.ylabel(r"$G_4(x)$")
    plt.title(r"$G_4$ (scaled, $s={}$)".format(s))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # compute overlap matrix using generic quadrature helper
    L = _default_L(centers, b, s=s)
    dx = s / 200
    xs = np.arange(-L, L + dx, dx, dtype=float)
    basis_funcs = build_basis(centers, b, s=s)
    S = overlap_matrix(basis_funcs, xs)
    print("Overlap matrix shape:", S.shape)
    np.set_printoptions(precision=8, suppress=True)
    print(S)
