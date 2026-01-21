import numpy as np
from gausslet import gausslet_scaled
from tools import make_centers_and_scale



# Coulomb kernel for e-e interaction
def coulomb_kernel_1d(x, xp, softening=1e-2):
    """
        1 / sqrt((x - xp)^2 + softening^2)
    """
    x  = np.asarray(x,  dtype=float)
    xp = np.asarray(xp, dtype=float)
    X  = x[:,  None]   # (Nx, 1)
    Xp = xp[None, :]   # (1, Nxp)
    return 1.0 / np.sqrt((X - Xp) ** 2 + softening ** 2)

# Create grid
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
        Phi[i, :] = gausslet_scaled(x, b, center=centers[i], s=scales)
    return Phi



def build_kinetic_matrix(Phi, dx):
    """
    T_ij = < phi_i | -1/2 d^2/dx^2 | phi_j >
    
    Using numerical gradients on the grid.
    
    Parameters:
        Phi : Phi[i, a]
        dx  : float, grid spacing
    
    Returns:
        T   : T[i, j] = Phi''[i, a] Phi[j, a]
    """
    # Calculate derivative dPhi/dx along the grid axis (axis 1, a)
    dPhi = np.gradient(Phi, dx, axis=1)
    d2Phi = np.gradient(dPhi, dx, axis=1)
    
    # Integrate: -0.5 * sum( Phi[i, x] * d2Phi[j, x] ) * dx
    # Using einsum: i (basis 1), j (basis 2), a (grid points)
    T = -0.5 * np.einsum('ia, ja -> ij', Phi, d2Phi) * dx
    
    return T


def build_external_potential_matrix(Phi, V_x, dx):
    """
    V_ext_ij = < phi_i | V(x) | phi_j >
             = integral dx phi_i(x) * V(x) * phi_j(x)
                 
    Parameters:
        Phi : Phi[i, a]
        V_x : V[a]
        dx  : float
        
    Returns:
        V_mat : V[i, j] = Phi[i, a] V[a] Phi[j, a]
    """
    # V_x needs to be broadcasted or multiplied along the grid axis 'a'
    # sum_{a} Phi[i, a] * V_x[a] * Phi[j, a] * dx
    V_mat = np.einsum('ia, a, ja -> ij', Phi, V_x, Phi) * dx
    return V_mat


def build_coulomb_tensor(Phi, x, dx, softening=1e-2):
    """
    V_ijkl = integral dx dx' phi_i(x) phi_j(x)
             (1 / sqrt((x - x')^2 + softening^2))
             phi_k(x') phi_l(x')

    Parameters:
        Phi      : Phi[i, a]
        x        : x[a]
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


def compute_coef(L, N, b, n_grid=500, softening=1e-2, file_name=None, coulomb=True, info=True):
    import os
    if not os.path.exists('data'):
        os.makedirs('data')

    centers, scales = make_centers_and_scale(L, N)
    centers.sort()
    x = np.linspace(-L, L, n_grid)
    dx = x[1] - x[0]
    
    Phi = build_basis_values(x, b, centers, scales)

    T = build_kinetic_matrix(Phi, dx)

    V_ext_func = 0.5 * x**2
    V_ext = build_external_potential_matrix(Phi, V_ext_func, dx)

    V_coulomb = None
    if coulomb:
        V_coulomb = build_coulomb_tensor(Phi, x, dx, softening=softening)
    save_path = 'data/' + file_name + '.npz'
    np.savez(save_path, 
             x=x, 
             T=T, 
             V_ext=V_ext, 
             V_coulomb=V_coulomb)
    
    if info:
        print("\n--- Summary ---")
        print(f"Kinetic Matrix T shape: {T.shape}")
        print(f"Potential Matrix V_ext shape: {V_ext.shape}")
        print(f"Coulomb Tensor V_coulomb shape: {V_coulomb.shape}")
        print(f"T[0,0] (Kinetic diagonal): {T[0,0]:.6f}")
        print("Done.")

if __name__ == "__main__":
    coeff_file = 'coefficient/G4.csv'
    _, b = np.loadtxt(coeff_file, unpack=True, delimiter=',')

    L_N_list = ((0.1, 51), (1, 51), (3, 51), (5, 51), (10, 51), (20, 51))

    for L, N in L_N_list:
        compute_coef(L, N, b, file_name=f'L{str(L).replace('.', '')}_N{str(N)}')

    

