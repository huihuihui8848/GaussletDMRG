# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import FermionSite
from tenpy.models.lattice import Chain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg


def load_integrals(npz_path='data/hamiltonian_integrals.npz'):
    data = np.load(npz_path)
    T = data['T']
    V_ext = data['V_ext']
    V_coul = data['V_coulomb']
    h = 0.5 * ((T + V_ext) + (T + V_ext).T)
    N = h.shape[0]
    U = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j:
                U[i, j] = V_coul[i, i, j, j]
    U = 0.5 * (U + U.T)
    np.fill_diagonal(U, 0.0)
    return h, U


class SpinlessFermionModel(CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        return FermionSite(conserve=conserve)

    def init_lattice(self, model_params):
        L = model_params['L']                       # 必填
        bc = model_params.get('bc', 'open')
        bc_MPS = model_params.get('bc_MPS', 'finite')
        site = self.init_sites(model_params)        # 不访问 self.sites，直接本地建
        return Chain(L, site, bc=bc, bc_MPS=bc_MPS)

    def init_terms(self, model_params):
        h = model_params['h']                       # (N,N)
        U = model_params['U']                       # (N,N)
        cutoff = model_params.get('cutoff', 1e-12)
        L = h.shape[0]

        # on-site: h_ii n_i
        for i in range(L):
            val = float(h[i, i])
            if abs(val) > cutoff:
                # 注意：特定“位置 i” → add_onsite_term（不是 add_onsite）
                self.add_onsite_term(val, i, 'N')

        # hopping: h_ij (c_i^† c_j + h.c.)，费米子要 JW 串
        for i in range(L):
            for j in range(i + 1, L):
                hij = float(h[i, j])
                if abs(hij) > cutoff:
                    # 参数顺序：strength, i, j, op_i, op_j
                    self.add_coupling_term(hij, i, j, 'Cd', 'C',
                                           op_string='JW', plus_hc=True)

        # # density-density: U_ij n_i n_j
        # for i in range(L):
        #     for j in range(i + 1, L):
        #         uij = float(U[i, j])
        #         if abs(uij) > cutoff:
        #             self.add_coupling_term(uij, i, j, 'N', 'N')


def run_dmrg(npz_path='data/hamiltonian_integrals.npz', electrons=10):
    h, U = load_integrals(npz_path)
    L = h.shape[0]
    assert electrons <= L
    print(f"L = {L}  |  ||h||₁ = {np.sum(np.abs(h)):.6g}  ||U||₁ = {np.sum(np.abs(U)):.6g}")

    model_params = dict(
        L=L, h=h, U=U,
        bc='open', bc_MPS='finite',
        conserve='N', cutoff=1e-12
    )
    M = SpinlessFermionModel(model_params)

    # 半填充初态（无自旋：0/1 占据）
    product_state =  ['empty'] * ((L-electrons)//2) + ['full'] * electrons + ['empty'] * ((L-electrons+1)//2)
    psi = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
    print(psi.expectation_value('N'))

    # 初态检查
    E_init = M.H_MPO.expectation_value(psi)
    print("⟨H⟩(init) =", float(E_init))
    print("∑⟨n_i⟩(init) =", float(np.sum(psi.expectation_value('N'))), "/", L)

    dmrg_params = {
        'mixer': True,
        'mixer_params': {'amplitude': 1e-3, 'decay': 1.01},
        'trunc_params': {'chi_min': 16, 'chi_max': 256, 'svd_min': 1e-10},
        'max_sweeps': 40,
        'min_sweeps': 10,
        'max_E_err': 1e-9,
        'verbose': True,
    }
    engine = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    E0, psi = engine.run()
    print("DMRG ground state energy E0 =", float(E0))
    print("∑⟨n_i⟩(GS) =", float(np.sum(psi.expectation_value('N'))), "/", L)
    print('chi = ', psi.chi)
    print(psi.expectation_value('N'))
    return float(E0), psi.expectation_value('N')


if __name__ == '__main__':
    max_N = 10
    E = np.zeros(max_N)
    all_densities = []
    L = None

    for N in range(max_N):
        E_N, n_i = run_dmrg('data/hamiltonian_integrals.npz', N)
        E[N] = E_N
        all_densities.append(n_i)
        if L is None:
            L = len(n_i)

    print("E(N) =", E)

    sites = np.arange(L)
    plt.figure()
    for N in range(max_N):
        plt.plot(sites, all_densities[N],
                 label=f"N={N}, E={E[N]:.6f}")
    plt.xlabel("site index i")
    plt.ylabel(r"$\langle n_i \rangle$")
    plt.title("Site occupation probability for different N")
    plt.legend()
    plt.tight_layout()
    plt.show()
    

    
