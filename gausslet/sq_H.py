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


# ---------- Quadrature & layout ----------
def legendre_integrate(func, a, b, npts=180):
    xg, wg = leggauss(npts)
    xp = 0.5 * (b - a) * xg + 0.5 * (a + b)
    fp = func(xp)
    return 0.5 * (b - a) * np.sum(wg * fp)

def centers_lin(a, b, N):
    return np.linspace(a, b, N)  # switch to midpoints if desired

# ---------- Matrix elements ----------
def overlap_element(i, j, a, b, N, coeffs_b, s, nquad=180):
    centers = centers_lin(a, b, N)
    ci, cj = centers[i], centers[j]
    def integrand(x):
        pi = gausslet_scaled(x, coeffs_b, center=ci, s=s)
        pj = gausslet_scaled(x, coeffs_b, center=cj, s=s)
        return pi * pj
    return legendre_integrate(integrand, a, b, npts=nquad)

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

def build_matrices(a, b, N, coeffs_b, s, V_callable, mass=1.0, hbar=1.0,
                   nquad=180, verbose=True):
    T = np.empty((N, N))
    Vmat = np.empty((N, N))
    S = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            S[i, j]   = overlap_element (i, j, a, b, N, coeffs_b, s, nquad)
            T[i, j]   = kinetic_element (i, j, a, b, N, coeffs_b, s, mass, hbar, nquad)
            Vmat[i, j]= potential_element(i, j, a, b, N, coeffs_b, s, V_callable, nquad)
    H = T + Vmat
    if verbose:
        condS = np.linalg.cond(S)
        print(f"[build] S shape={S.shape}, cond(S)≈{condS:.3e}")
    return S, T, Vmat, H

# ---------- Löwdin orthogonalization ----------
def lowdin_orthogonalize(S, H, eps=1e-12):
    """
    Symmetric Löwdin: S = U diag(s) U^T,  S^{-1/2} = U diag(s^{-1/2}) U^T.
    Returns h_ortho = S^{-1/2} H S^{-1/2} and the transform X=S^{-1/2}.
    """
    s, U = np.linalg.eigh(S)
    # filter near-zero modes to avoid blowup
    keep = s > eps * s.max()
    if not np.all(keep):
        print(f"[lowdin] dropping {np.size(s)-np.count_nonzero(keep)} near-null modes")
    s_kept = s[keep]
    U_kept = U[:, keep]
    Sinvhalf = (U_kept / np.sqrt(s_kept)) @ U_kept.T
    h_ortho = Sinvhalf @ H @ Sinvhalf
    # symmetrize numerically
    h_ortho = 0.5 * (h_ortho + h_ortho.T.conj())
    return h_ortho, Sinvhalf

# ---------- Extraction helpers ----------
def extract_tij(h_ortho, centers=None, magnitude_tol=1e-8, max_distance=None):
    """
    From the orthonormal one-body matrix h_ortho, return:
      - onsite energies eps[i] = h_ortho[i,i]
      - hoppings list of (i, j, t_ij, |r_i - r_j|) for i<j with |t_ij|>tol
        and (optionally) |r_i-r_j| <= max_distance.
    """
    N = h_ortho.shape[0]
    eps = np.real(np.diag(h_ortho).copy())
    hops = []
    for i in range(N):
        for j in range(i+1, N):
            t = h_ortho[i, j]
            if np.abs(t) < magnitude_tol:
                continue
            dist = None
            if centers is not None:
                dist = abs(centers[i] - centers[j])
                if (max_distance is not None) and (dist > max_distance):
                    continue
            hops.append((i, j, np.real_if_close(t), dist))
    # Sort by distance then magnitude
    hops.sort(key=lambda rec: (np.inf if rec[3] is None else rec[3], -abs(rec[2])))
    return eps, hops

def pretty_print_second_quant(eps, hops, spin_degenerate=False):
    """
    Print H = sum_i eps_i n_i + sum_{i<j} (t_ij c_i^† c_j + h.c.)
    If spin_degenerate=True, the same eps,t apply for σ in {↑,↓}.
    """
    print("One-body second-quantized form:")
    if spin_degenerate:
        print("  H = Σ_i ε_i Σ_σ n_{iσ} + Σ_{i<j} Σ_σ (t_{ij} c_{iσ}† c_{jσ} + h.c.)")
    else:
        print("  H = Σ_i ε_i n_i + Σ_{i<j} (t_{ij} c_i† c_j + h.c.)")
    print("\nOn-site energies ε_i:")
    for i, e in enumerate(eps):
        print(f"  i={i:3d}  ε={e:+.8e}")
    print("\nHopping amplitudes t_{ij} (i<j):")
    for (i, j, t, dist) in hops:
        dstr = "" if dist is None else f"  |Δx|={dist:.6g}"
        val  = complex(t)  # ensure complex formatting if needed
        print(f"  ({i:3d},{j:3d})  t={val.real:+.8e}{val.imag:+.8e}j{dstr}")

# ---------- Example usage ----------
if __name__ == "__main__":
    # Domain & basis
    a, b = -10.0, 10.0
    N = 41
    dx = (b - a) / (N - 1)
    s = dx  # tune for locality/conditioning

    # Replace with your pre-defined Gausslet coefficients
    _, b_coeffs = np.loadtxt('coefficient/G4.csv', unpack=True, delimiter=',')

    # Potential
    def V(x):
        return 0.5 * x**2  # harmonic oscillator example

    mass = 1.0
    hbar = 1.0

    # Build matrices
    S, T, Vmat, H = build_matrices(a, b, N, b_coeffs, s, V, mass=mass, hbar=hbar, nquad=220)

    # Löwdin orthogonalization -> orthonormal one-body matrix
    h_ortho, X = lowdin_orthogonalize(S, H, eps=1e-12)

    # Centers (for optional distance cut)
    centers = centers_lin(a, b, N)

    # Extract onsite & hoppings (truncate by magnitude and/or real-space range)
    eps, hops = extract_tij(
        h_ortho,
        centers=centers,
        magnitude_tol=1e-7,     # ignore ultra-small couplings
        max_distance=3.5 * dx   # e.g., keep up to ~3-4 neighbor shells
    )

    # Print second-quantized form
    pretty_print_second_quant(eps, hops, spin_degenerate=True)

    # Optionally save to disk
    # np.savez('tight_binding_one_body.npz',
    #          eps=eps, hops=np.array(hops, dtype=object),
    #          h_ortho=h_ortho, centers=centers, X=X, S=S, H=H)
