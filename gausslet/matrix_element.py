import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import leggauss
from scipy.sparse import dok_matrix, csr_matrix

# ---------- Your basis primitives ----------
def gaussian(x, mu=0.0, sigma=1.0):
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

# ---------- Analytic derivatives (stable) ----------
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

def _gausslet_primary_second_derivative(u, b):
    u = np.atleast_1d(u)
    x3 = 3.0 * u
    b = np.asarray(b, dtype=float)
    j = np.arange(len(b), dtype=float)

    def g2(z, mu):
        g = gaussian(z, mu, 1.0)
        return ((z - mu) ** 2 - 1.0) * g
    res = 9.0 * b[0] * g2(x3, j[0])
    if j.size > 1:
        Gp = g2(x3[:, None],  j[1:])
        Gn = g2(x3[:, None], -j[1:])
        res = res + 9.0 * (Gp @ b[1:] + Gn @ b[1:])
    return np.squeeze(res)

def gausslet_scaled_prime(x, b, center=0.0, s=1.0):
    x = np.asarray(x, dtype=float)
    u = (x - center) / s
    return (1.0 / (s ** 1.5)) * _gausslet_primary_first_derivative(u, b)

def gausslet_scaled_second(x, b, center=0.0, s=1.0):
    x = np.asarray(x, dtype=float)
    u = (x - center) / s
    return (1.0 / (s ** 2.5)) * _gausslet_primary_second_derivative(u, b)

# ---------- Quadrature & layout ----------
def legendre_integrate(func, a, b, npts=180):
    xg, wg = leggauss(npts)
    xp = 0.5 * (b - a) * xg + 0.5 * (a + b)
    fp = func(xp)
    return 0.5 * (b - a) * np.sum(wg * fp)

def centers_lin(a, b, N):
    return np.linspace(a, b, N)

# ---------- Matrix elements ----------
def kinetic_element(i, j, a, b, N, coeffs_b, s, mass, hbar=1.0, nquad=180):
    centers = centers_lin(a, b, N)
    ci, cj = centers[i], centers[j]
    def integrand(x):
        dpi = gausslet_scaled_prime(x, coeffs_b, center=ci, s=s)
        dpj = gausslet_scaled_prime(x, coeffs_b, center=cj, s=s)
        return dpi * dpj
    pref = (hbar ** 2) / (2.0 * mass)
    return pref * legendre_integrate(integrand, a, b, npts=nquad)

def potential_element(i, j, a, b, N, coeffs_b, s, V_callable, nquad=180):
    centers = centers_lin(a, b, N)
    ci, cj = centers[i], centers[j]
    def integrand(x):
        pi = gausslet_scaled(x, coeffs_b, center=ci, s=s)
        pj = gausslet_scaled(x, coeffs_b, center=cj, s=s)
        return pi * V_callable(x) * pj
    return legendre_integrate(integrand, a, b, npts=nquad)

def overlap_element(i, j, a, b, N, coeffs_b, s, nquad=180):
    centers = centers_lin(a, b, N)
    ci, cj = centers[i], centers[j]
    def integrand(x):
        pi = gausslet_scaled(x, coeffs_b, center=ci, s=s)
        pj = gausslet_scaled(x, coeffs_b, center=cj, s=s)
        return pi * pj
    return legendre_integrate(integrand, a, b, npts=nquad)

# ---------- NEW: sparse builders with threshold ----------
def build_sparse_matrix(N, element_fn, tol=1e-12):
    """
    element_fn(i, j) -> float
    Only stores entries with |value| >= tol
    """
    M = dok_matrix((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            val = element_fn(i, j)
            if abs(val) >= tol:
                M[i, j] = val
    return M.tocsr()

# ---------- Visualization helpers ----------
def plot_matrix_heatmap(M, title):
    # Accepts dense or sparse; converts if needed
    if hasattr(M, "toarray"):
        A = M.toarray()
    else:
        A = np.asarray(M, dtype=float)
    plt.figure(figsize=(5, 4))
    plt.imshow(A, origin='upper', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('j')
    plt.ylabel('i')
    plt.tight_layout()
    plt.show()

def band_decay_stats(M, tol=1e-12, title=""):
    """
    Works for dense or CSR/CSC sparse matrices.
    Returns:
      band_max: array of max |M[i,i+k]| for k=0..N-1
      k_first_all_below: smallest k such that for all k' >= k, band_max[k'] < tol
                         (None if no such k)
    Also plots band_max on semilogy.
    """
    if hasattr(M, "shape"):
        N = M.shape[0]
    else:
        M = np.asarray(M)
        N = M.shape[0]

    # Ensure CSR for .diagonal(k) on sparse
    if hasattr(M, "tocsr"):
        Mcsr = M.tocsr()
        diag_fn = lambda kk: Mcsr.diagonal(kk)
    else:
        A = np.asarray(M, dtype=float)
        diag_fn = lambda kk: np.diag(A, k=kk)

    ks = np.arange(N)
    band_max = np.zeros(N, dtype=float)
    for k in ks:
        d = diag_fn(k)
        band_max[k] = 0.0 if d.size == 0 else np.max(np.abs(d))

    # Find smallest k so that all remaining bands are < tol
    # Equivalently: last index with >= tol, then +1
    above = band_max >= tol
    if not np.any(above):
        k_first_all_below = 0
    else:
        last_sig = np.where(above)[0].max()
        k_first_all_below = last_sig + 1 if last_sig + 1 < N else None

    # Plot
    plt.figure(figsize=(5.2, 3.6))
    plt.semilogy(ks, band_max, marker='o', linestyle='-')
    plt.title((title + " ").strip() + "(max |band k|)")
    plt.xlabel('|i - j| = k')
    plt.ylabel('max |M[i,i+k]|')
    plt.tight_layout()
    plt.show()

    # Print the cutoff summary
    if k_first_all_below is None:
        print(f"{title} No finite k where all remaining bands fall below {tol:.1e}.")
    else:
        print(f"{title} Smallest k with all subsequent bands < {tol:.1e}: k = {k_first_all_below}")
    return band_max, k_first_all_below

# ---------- Example main ----------
if __name__ == '__main__':
    # Domain & basis
    a, b = -8.0, 8.0
    Ns = [30]

    # Load gausslet coefficient vector (shape like your G4.csv second column)
    # Make sure 'coefficient/G4.csv' exists; otherwise replace this line with your coeffs.
    _, b_coeffs = np.loadtxt('coefficient/G4.csv', unpack=True, delimiter=',')

    # Physical params
    m = 1.0
    hbar = 1.0

    # Potential
    def V(x):
        return 0.5 * x**2  # harmonic oscillator

    # Truncation threshold for sparsity
    TOL = 1e-12
    NQUAD = 180
    for N in Ns:
        dx = (b - a) / (N - 1)
        s = dx
        centers = centers_lin(a, b, N)

        # ---------- Build sparse matrices with truncation ----------
        T = build_sparse_matrix(
            N,
            lambda i, j: kinetic_element(i, j, a, b, N, b_coeffs, s, m, hbar, nquad=NQUAD),
            tol=TOL
        )
        Vmat = build_sparse_matrix(
            N,
            lambda i, j: potential_element(i, j, a, b, N, b_coeffs, s, V, nquad=NQUAD),
            tol=TOL
        )
        S = build_sparse_matrix(
            N,
            lambda i, j: overlap_element(i, j, a, b, N, b_coeffs, s, nquad=NQUAD),
            tol=TOL
        )
        H = (T + Vmat).tocsr()

        # ---------- Band-decay diagnostics & cutoff k ----------
        band_S, kcut_S = band_decay_stats(S, tol=TOL, title="Overlap S:")
        band_H, kcut_H = band_decay_stats(H, tol=TOL, title="Hamiltonian H:")

        print("k* (S) =", kcut_S, 'with N =', N)
        print("k* (H) =", kcut_H, 'with N =', N)

    # ---------- Plots (convert sparse->dense just for visualization) ----------
    # plot_matrix_heatmap(S, 'Overlap matrix S (sparse)')
    # plot_matrix_heatmap(T, 'Kinetic matrix T (sparse)')
    # plot_matrix_heatmap(Vmat, 'Potential matrix V (sparse)')
    # plot_matrix_heatmap(H, 'Hamiltonian H (sparse)')



