import numpy as np

from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import FermionSite
from tenpy.models.lattice import Chain

from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS



def load_integrals(npz_path='data/hamiltonian_integrals.npz'):
    """
    从 Gausslet 积分文件中读取:
        T        : (N, N)
        V_ext    : (N, N)
        V_coulomb: (N, N, N, N)

    构造:
        h_ij = T_ij + V_ext_ij   (一体项)
        U_ij ≈ V_{i i j j}       (density-density 直项)
    """
    data = np.load(npz_path)

    T = data['T']              # (N, N)
    V_ext = data['V_ext']      # (N, N)
    V_coul = data['V_coulomb'] # (N, N, N, N)

    # 一体项: h = T + V_ext，并数值对称化
    h = T + V_ext
    h = 0.5 * (h + h.T)

    N = h.shape[0]

    # 两体密度-密度: U_ij ≈ V_{i i j j}
    U = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            U[i, j] = V_coul[i, i, j, j]

    # 对称化 U，去掉对角
    U = 0.5 * (U + U.T)
    np.fill_diagonal(U, 0.0)

    return h, U



class SpinlessFermionModel(CouplingMPOModel):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'N')
        return FermionSite(conserve=conserve)

    def init_lattice(self, model_params):
        # 关键：不要用 self.sites，这里直接拿本地 site
        L = model_params['L']
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = model_params.get('bc', 'open')
        site = self.init_sites(model_params)
        return Chain(L, site, bc=bc, bc_MPS=bc_MPS)

    def init_terms(self, model_params):
        h = model_params['h']  # (N, N)
        U = model_params['U']  # (N, N)
        N = h.shape[0]

        # on-site: h_ii n_i
        for i in range(N):
            hii = float(h[i, i])
            if abs(hii) > 1e-12:
                self.add_onsite_term(hii, i, 'N')

        # hopping: h_ij (c_i^† c_j + h.c.), 费米子需 JW 串
        for i in range(N):
            for j in range(i + 1, N):
                hij = float(h[i, j])
                if abs(hij) > 1e-12:
                    # 正确顺序：strength, i, j, opname_i, opname_j
                    self.add_coupling_term(hij, i, j, 'Cd', 'C',
                                        op_string='JW', plus_hc=True)

        # density-density: U_ij n_i n_j
        for i in range(N):
            for j in range(i + 1, N):
                uij = float(U[i, j])
                if abs(uij) > 1e-12:
                    # 正确顺序：strength, i, j, opname_i, opname_j
                    self.add_coupling_term(uij, i, j, 'N', 'N')


# 读取积分数据
h, U = load_integrals('data/hamiltonian_integrals.npz')
N = h.shape[0]

# 构造模型参数
model_params = {
    'L': N,
    'h': h,
    'U': U,
    'bc_MPS': 'finite',   # 有限 MPS 边界条件（开边界）
    'bc': 'open',         # 开放的链式边界（不考虑环绕耦合）
    'conserve': 'N'       # 保持粒子数守恒以固定粒子数
}
M = SpinlessFermionModel(model_params)

# 构造初始产品态：一半站点占据（确保总粒子数=N/2）
num_particles = N // 2  # 若N为奇数，可取 floor(N/2) 或相邻整数作为近似半填充
product_state = ['full'] * num_particles + ['empty'] * (N - num_particles)
# 为避免初态局域化，可选择交替填充: e.g., product_state = ['full','empty'] * (N//2)

# 创建初始 MPS (有限 MPS)

psi0 = MPS.from_product_state(M.lat.mps_sites(), product_state, bc=M.lat.bc_MPS)
print("⟨n_i⟩ =", psi0.expectation_value('N'))



dmrg_params = {
    'mixer': True,  # 帮助跳出local minima
    'mixer_params': {
        'amplitude': 1e-3,
        'decay': 1.01,
    },
    'trunc_params': {
        'chi_max': 100,   # 不够再往上加
        'svd_min': 1e-9,
    },
    'max_sweeps': 1000,
    'max_E_err': 1e-8,
}

# 初始化并运行 DMRG 引擎
engine = dmrg.TwoSiteDMRGEngine(psi0, M, dmrg_params)
E, psi = engine.run()  # 执行DMRG迭代，返回基态能量 E 和基态 MPS psi
print("Ground state energy =", E)

print("⟨n_i⟩ =", psi.expectation_value('N'))
