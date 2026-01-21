# gausslet_second_quant.py
# Build {S,T,V,H} in a Gausslet basis, orthogonalize with Löwdin S^{-1/2},
# and extract second-quantized matrix elements: H = sum_{ij} t_ij c_i^† c_j

import numpy as np
from numpy.polynomial.legendre import leggauss

import numpy as np

# ---------- Gausslet primitives ----------
def gaussian(x, mu=0.0, sigma=1.0):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = float(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gausslet_primary(x, b):
    """
    Primary 'gausslet' as a symmetric sum of unit-variance Gaussians
    placed at integer nodes with 3x coordinate scaling (as in your snippet).
    """
    x = np.asarray(x, dtype=float)
    x = np.atleast_1d(x)
    x3 = 3.0 * x
    b = np.asarray(b, dtype=float)
    j = np.arange(len(b), dtype=float)

    # center term
    res = b[0] * gaussian(x3, j[0], sigma=1.0)
    # symmetric side terms
    if j.size > 1:
        Gpos = gaussian(x3[:, None],  j[1:], sigma=1.0)
        Gneg = gaussian(x3[:, None], -j[1:], sigma=1.0)
        res = res + Gpos @ b[1:] + Gneg @ b[1:]
    return np.squeeze(res)

def gausslet_scaled(x, b, center=0.0, s=1.0):
    """
    Scaled/shifted gausslet with L2-preserving prefactor 1/sqrt(s).
    """
    x = np.asarray(x, dtype=float)
    y = (1.0 / np.sqrt(s)) * gausslet_primary((x - center) / s, b)
    return np.squeeze(y)

# Analytic derivatives (kept for completeness / future kinetic-energy work)
def _gausslet_primary_first_derivative(u, b):
    u = np.atleast_1d(u)
    x3 = 3.0 * u
    b = np.asarray(b, dtype=float)
    j = np.arange(len(b), dtype=float)
    def gprime(z, mu):
        g = gaussian(z, mu, 1.0)
        return -(z - mu) * g
    res = 3.0 * b[0] * gprime(x3, j[0])
    if j.size > 1:
        Gp = gprime(x3[:, None],  j[1:])
        Gn = gprime(x3[:, None], -j[1:])
        res = res + 3.0 * (Gp @ b[1:] + Gn @ b[1:])
    return np.squeeze(res)

def gausslet_scaled_prime(x, b, center=0.0, s=1.0):
    x = np.asarray(x, dtype=float)
    u = (x - center) / s
    return (1.0 / (s ** 1.5)) * _gausslet_primary_first_derivative(u, b)

# ---------- Soft-Coulomb kernel (1D) ----------
def soft_coulomb(r, eta):
    """
    1D soft-Coulomb: V(r) = 1 / sqrt(r^2 + eta^2), with eta > 0 to regularize.
    """
    r = np.asarray(r, dtype=float)
    return 1.0 / np.sqrt(r*r + float(eta)**2)

# ---------- Build Gausslet basis on a uniform grid ----------
def build_basis_1d(centers, b, s, x):
    """
    Returns Phi[i, x] = phi_i(x) normalized on the grid (trapezoidal).
    """
    centers = np.asarray(centers, dtype=float)
    N = centers.size
    x = np.asarray(x, dtype=float)

    Phi = np.empty((N, x.size), dtype=float)
    for i, c in enumerate(centers):
        Phi[i] = gausslet_scaled(x, b, center=c, s=s)

    # Normalize on the grid with trapezoidal rule
    norms = np.sqrt(np.trapz(Phi**2, x, axis=1))
    Phi /= norms[:, None]
    return Phi

# ---------- Two-electron integral assembly ----------
def compute_coulomb_tensor(Phi, x, eta):
    """
    Compute V_{ijkl} = \int dx \int dx' phi_i(x) phi_k(x) * V(x-x') * phi_j(x') phi_l(x')
    using matrix contraction with the kernel K(x,x') = soft_coulomb(x-x', eta).

    Returns V as shape (N, N, N, N).
    """
    x = np.asarray(x, dtype=float)
    dx = x[1] - x[0]

    # Kernel K[x, x'] = V(x - x')
    dx_grid = x[:, None] - x[None, :]
    K = soft_coulomb(dx_grid, eta)

    # Include integration weights dx^2 in the kernel so contractions are simple sums
    K = K * (dx * dx)

    N, Ng = Phi.shape
    # Pair functions A[(i,k), x] = phi_i(x)*phi_k(x)
    pairs = [(i, k) for i in range(N) for k in range(N)]
    A = np.empty((N*N, Ng), dtype=float)
    for p, (i, k) in enumerate(pairs):
        A[p] = Phi[i] * Phi[k]

    # Contract: G[p, q] = A K A^T  => V_{i j k l} with p=(i,k), q=(j,l)
    G = A @ (K @ A.T)

    # Reshape G -> V[i, j, k, l]
    V = np.empty((N, N, N, N), dtype=float)
    for p, (i, k) in enumerate(pairs):
        for q, (j, l) in enumerate(pairs):
            V[i, j, k, l] = G[p, q]
    return V

# ---------- Convenience wrapper ----------
def compute_and_save_V_npz(
    L=20.0,
    Ng=2048,
    N=20,
    b=None,
    s=None,
    eta=0.2,
    out_path="data/V_ijkl_softcoulomb_gausslet_1D.npz",
):
    """
    Build a 1D Gausslet basis on a uniform grid and compute the two-electron
    soft-Coulomb tensor. Saves an NPZ with V, grid, basis and metadata.

    Args:
        L: box length (domain is [-L/2, L/2))
        Ng: number of grid points
        N: number of Gausslet centers (uniformly spaced)
        b: Gausslet coefficient vector; default = [1.0] (single Gaussian-like)
        s: Gausslet scale; default = 0.5 * (L/N)
        eta: soft-Coulomb regularization parameter (>0)
        out_path: NPZ path to save

    Returns:
        (V, Phi, x, centers) and saves NPZ to disk.
    """
    # Grid
    x = np.linspace(-L/2, L/2, int(Ng), endpoint=False)
    spacing = L / float(N)
    centers = np.linspace(-L/2 + 0.5*spacing, L/2 - 0.5*spacing, int(N))

    # Defaults for gausslet parameters
    if b is None:
        b = np.array([1.0], dtype=float)
    else:
        b = np.asarray(b, dtype=float)
    if s is None:
        s = 0.5 * spacing

    # Basis
    Phi = build_basis_1d(centers, b, s, x)

    # Two-electron tensor
    V = compute_coulomb_tensor(Phi, x, eta)

    # Save everything
    np.savez_compressed(
        out_path,
        V=V,
        x=x,
        centers=centers,
        Phi=Phi,
        L=np.array([L]),
        Ng=np.array([Ng]),
        spacing=np.array([spacing]),
        s=np.array([s]),
        b=b,
        eta=np.array([eta]),
        note=np.array([
            "V_ijkl = <phi_i,phi_j| 1/sqrt((x-x')^2+eta^2) |phi_k,phi_l> (1D soft Coulomb), "
            "Gausslet basis normalized on grid; dx^2 included in kernel K."
        ]),
    )
    return V, Phi, x, centers

# ---------- Example usage ----------
if __name__ == "__main__":
    # Adjust these as needed
    L = 20.0
    Ng = 2048
    N = 12
    _, b = np.loadtxt('coefficient/G4.csv', unpack=True, delimiter=',')
    eta = 0.2              # softening length (regularizes 1D Coulomb)

    # By default s = 0.5 * (L/N); pass explicitly if you want a different width
    V, Phi, x, centers = compute_and_save_V_npz(
        L=L, Ng=Ng, N=N, b=b, s=L/N, eta=eta
    )

    # Optional: quick symmetry sanity checks
    # V_{ijkl} should satisfy permutational symmetries like V_{ijkl} = V_{jilk} = V_{klij}
    i, j, k, l = 0, 1, 0, 1
    print("V[i,j,k,l] =", V[i,j,k,l])
    print("V[j,i,l,k] =", V[j,i,l,k])
    print("V[k,l,i,j] =", V[k,l,i,j])
    print("Saved to:", "data/V_ijkl_softcoulomb_gausslet_1D.npz")


