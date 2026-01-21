import numpy as np
import os

####################################
# Gausslet Definitions
####################################

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

####################################
# Coulomb kernel (Soft)
####################################

def coulomb_kernel_1d(x, xp, softening=1e-2):
    """
        1 / sqrt((x - xp)^2 + softening^2)
    """
    x  = np.asarray(x,  dtype=float)
    xp = np.asarray(xp, dtype=float)
    X  = x[:,  None]   # (Nx, 1)
    Xp = xp[None, :]   # (1, Nxp)
    return 1.0 / np.sqrt((X - Xp) ** 2 + softening ** 2)

####################################
# Build Basis Values on Grid
####################################

def build_basis_values(x, b, centers, scales):
    """
    Calculate a set of gausslet basis functions on the given grid x:
        phi_i(x) = gausslet_scaled(x, b, center_i, scale_i)
    
    Parameters:
        x       : 1D numpy array, real space grid
        b       : Coefficients array for gausslet_primary
        centers : Array of center positions for each basis, length = n_basis
        scales  : Array of scales s_i for each basis, length = n_basis
        
    Returns:
        Phi: shape = (n_basis, Nx)
             Phi[i, a] = phi_i(x[a])
    """
    x = np.asarray(x, dtype=float)
    centers = np.asarray(centers, dtype=float)
    scales  = np.asarray(scales,  dtype=float)

    n_basis = centers.size
    Nx = x.size

    Phi = np.empty((n_basis, Nx), dtype=float)
    for i in range(n_basis):
        Phi[i, :] = gausslet_scaled(x, b, center=centers[i], s=scales[i])
    return Phi

####################################
# 1. Calculation: Kinetic Energy T_ij
####################################

def build_kinetic_matrix(Phi, dx):
    """
    Numerically calculate the Kinetic Energy Matrix:
        T_ij = < phi_i | -1/2 d^2/dx^2 | phi_j >
    
    Using numerical gradients on the grid.
    
    Parameters:
        Phi : shape (n_basis, Nx), basis values on grid
        dx  : float, grid spacing
    
    Returns:
        T   : shape (n_basis, n_basis)
    """
    # Calculate first derivative dPhi/dx along the grid axis (axis 1)
    dPhi = np.gradient(Phi, dx, axis=1)
    
    # Calculate second derivative d^2Phi/dx^2
    d2Phi = np.gradient(dPhi, dx, axis=1)
    
    # Integrate: -0.5 * sum( Phi[i, x] * d2Phi[j, x] ) * dx
    # Using einsum: i (basis 1), j (basis 2), a (grid points)
    T = -0.5 * np.einsum('ia, ja -> ij', Phi, d2Phi) * dx
    
    return T

####################################
# 2. Calculation: External Potential V_ij
####################################

def build_external_potential_matrix(Phi, V_x, dx):
    """
    Calculate matrix elements for an arbitrary external potential:
        V_ext_ij = < phi_i | V(x) | phi_j >
                 = integral dx phi_i(x) * V(x) * phi_j(x)
                 
    Parameters:
        Phi : shape (n_basis, Nx)
        V_x : shape (Nx,), the values of the potential on the grid
        dx  : float
        
    Returns:
        V_mat : shape (n_basis, n_basis)
    """
    # V_x needs to be broadcasted or multiplied along the grid axis 'a'
    # sum_{a} Phi[i, a] * V_x[a] * Phi[j, a] * dx
    V_mat = np.einsum('ia, a, ja -> ij', Phi, V_x, Phi) * dx
    return V_mat

####################################
# 3. Calculation: Coulomb Tensor V_ijkl
####################################

def build_coulomb_tensor(Phi, x, dx, softening=1e-2):
    """
    Numerically calculate 1D Coulomb integrals:
        V_ijkl = integral dx dx' phi_i(x) phi_j(x)
                 (1 / sqrt((x - x')^2 + softening^2))
                 phi_k(x') phi_l(x')

    Parameters:
        Phi      : Basis values on grid (n_basis, Nx)
        x        : Real space grid (Nx,)
        dx       : Grid spacing
        softening: Softening parameter for the kernel

    Returns:
        V : Coulomb tensor, shape = (n_basis, n_basis, n_basis, n_basis)
    """
    n_basis = Phi.shape[0]

    # Construct Coulomb kernel K[a,b]
    K = coulomb_kernel_1d(x, x, softening=softening)  # shape (Nx, Nx)
    # print("Kernel shape:", K.shape)

    # Pre-calculate pair densities: phi_i(x)phi_j(x)
    # P[i,j,a] = phi_i(a) * phi_j(a)
    P = np.einsum('ia,ja->ija', Phi, Phi)   # (n_basis, n_basis, Nx)
    
    # Q is structurally identical to P, just different indices for the second electron
    Q = P  # Reference assignment (read-only usage is fine)

    # Compute double integral:
    # V_ijkl = sum_{a,b} P[i,j,a] * K[a,b] * Q[k,l,b] * dx^2
    V = np.einsum('ija,ab,klb->ijkl', P, K, Q) * dx * dx

    return V

def make_centers_and_scales(L, N):
    """
    Generate:
        centers: [0, -2s, 2s, -4s, 4s, ...] symmetrically arranged
        scales : same scale s = L / N for every basis function

    Requirement:
        N = 1 + 2*M must be odd.
    """
    if N % 2 == 0:
        raise ValueError(
            f"N = {N} is even. For symmetric arrangement 0, +/-2s, ..., "
            "the number of basis functions N must be odd (N = 1 + 2*M)."
        )

    s = L / N
    centers = np.zeros(N, dtype=float)

    M = (N - 1) // 2             # There are M pairs of +/-
    for i in range(1, M + 1):
        centers[2 * i - 1] = -2 * i * s   # Odd index: negative
        centers[2 * i]     =  2 * i * s   # Even index: positive

    scales = np.full(N, s, dtype=float)   # Same scale for all
    return centers, scales


if __name__ == "__main__":
    # Ensure data directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load coefficients (Dummy check if file exists, else use random for demo)
    coeff_file = 'coefficient/G4.csv'
    _, b = np.loadtxt(coeff_file, unpack=True, delimiter=',')

    # System parameters
    L = 1.0
    N = 31
    n_grid = 500
    softening = 1e-2
    
    # 1. Setup Grid and Basis
    centers, scales = make_centers_and_scales(L, N)
    centers.sort()
    x = np.linspace(-L, L, n_grid)
    dx = x[1] - x[0]
    
    print("Building Basis Values...")
    Phi = build_basis_values(x, b, centers, scales)
    print(f"Phi shape: {Phi.shape}")

    # 2. Compute Kinetic Energy Matrix (T)
    print("Computing Kinetic Energy Matrix...")
    T = build_kinetic_matrix(Phi, dx)
    
    # 3. Compute External Potential Matrix (V_ext)
    # Example: Harmonic Oscillator Potential V(x) = 0.5 * x^2
    print("Computing External Potential Matrix (Harmonic Oscillator)...")
    V_ext_func = 0.5 * x**2
    V_ext = build_external_potential_matrix(Phi, V_ext_func, dx)

    # 4. Compute Coulomb Interaction Tensor (V_coulomb)
    print("Computing Coulomb Interaction Tensor...")
    V_coulomb = build_coulomb_tensor(Phi, x, dx, softening=softening)

    # 5. Save Results
    save_path = 'data/hamiltonian_integrals.npz'
    print(f"Saving all results to {save_path}...")
    np.savez(save_path, 
             x=x, 
             T=T, 
             V_ext=V_ext, 
             V_coulomb=V_coulomb)
    
    # Print sample values to verify
    print("\n--- Summary ---")
    print(f"Kinetic Matrix T shape: {T.shape}")
    print(f"Potential Matrix V_ext shape: {V_ext.shape}")
    print(f"Coulomb Tensor V_coulomb shape: {V_coulomb.shape}")
    print(f"T[0,0] (Kinetic diagonal): {T[0,0]:.6f}")
    print("Done.")