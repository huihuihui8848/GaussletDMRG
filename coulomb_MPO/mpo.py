import numpy as np
import random

from tenpy.networks.site import FermionSite
from tenpy.models.lattice import Chain
from tenpy.networks.terms import ExponentiallyDecayingTerms
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
from tenpy.models.model import MPOModel

def fit_inverse_with_exponentials(
    N_max,
    K,
    mu_min=1e-3,
    mu_max=1.0,
    weight_power=1.0,
):
    """
    用 sum_k alpha_k * lambda_k^n 拟合 1/n, n=1,...,N_max

    参数:
        N_max        : 最大距离 n_max
        K            : 指数项个数
        mu_min, mu_max : 通过 lambda_k = exp(-mu_k) 生成一串 lambda_k,
                        mu_k 在 [mu_min, mu_max] 上 log 均匀
        weight_power : 拟合时的权重 w_n = n^weight_power
                       (可以调大一点让小 n 拟合更好)

    返回:
        lambdas : 形状 (K,) 的数组, lambda_k
        alphas  : 形状 (K,) 的数组, alpha_k
        n       : 1..N_max
        y       : 1/n
        y_fit   : 拟合后的值
    """
    # 距离 n = 1, 2, ..., N_max
    n = np.arange(1, N_max + 1, dtype=float)
    y = 1.0 / n

    # 生成 mu_k 和 lambda_k
    mus = np.logspace(np.log10(mu_min), np.log10(mu_max), K)
    lambdas = np.exp(-mus)  # 0<lambda_k<1

    # 构造矩阵 A_{n,k} = lambda_k^n
    # A 的形状是 (N_max, K)
    A = lambdas**n[:, None]

    # 权重: w_n = n^weight_power
    w = n**weight_power
    WA = A * w[:, None]   # 每一行乘对应权重
    Wy = y * w

    # 最小二乘解 alpha
    # WA @ alpha ≈ Wy
    alphas, residuals, rank, s = np.linalg.lstsq(WA, Wy, rcond=None)

    # 计算拟合结果
    y_fit = A @ alphas

    return lambdas, alphas, n, y, y_fit

def make_coulomb_mpo_from_fit(L, alphas, lambdas, bc='finite',
                              cutoff=1e-12):
    """
    用 sum_k alpha_k lambda_k^{|i-j|} 近似 1/|i-j|
    构造 H = sum_{i<j} (sum_k alpha_k lambda_k^{|i-j|}) n_i n_j 的 MPO.

    Parameters
    ----------
    L : int
        chian size
    alphas, lambdas : 1D array-like
        fit result
    bc : 'finite' or 'infinite'
    cutoff : float

    Returns
    -------
    H_mpo : tenpy.networks.mpo.MPO
        Coulomb MPO
    lat : tenpy.models.lattice.Chain
    """
    alphas = np.asarray(alphas, dtype=float)
    lambdas = np.asarray(lambdas, dtype=float)
    assert alphas.shape == lambdas.shape

    # 1. 定义 local Hilbert space：spinless fermion，有算符 'N' = number operator:contentReference[oaicite:1]{index=1}
    site = FermionSite(conserve='N')  # 你也可以改成 SpinHalfSite 等
    lat = Chain(L, site)
    sites = lat.mps_sites()

    # 2. 建立 ExponentiallyDecayingTerms 对象
    exp_terms = ExponentiallyDecayingTerms(L)

    # 3. 把每个 (alpha_k, lambda_k) 加进去：
    #    strength * sum_{i<j} lambda^{|i-j|} N_i N_j
    for a, lam in zip(alphas, lambdas):
        exp_terms.add_exponentially_decaying_coupling(
            strength=a,
            lambda_=lam,
            op_i='N',      # 左边算符名字
            op_j='N',      # 右边算符名字
            subsites=None,       # 默认为整个链
            subsites_start=None, # 1D 链用不到
            op_string='Id'      # N_i N_j 不需要 JW string
        )

    # 4. 先把这些长程项变成 TermList
    term_list = exp_terms.to_TermList(cutoff=cutoff, bc=bc)

    # 5. 用 MPOGraph.from_term_list -> build_MPO 得到真正的 MPO:contentReference[oaicite:3]{index=3}
    graph = MPOGraph.from_term_list(term_list, sites=sites, bc=bc,
                                    insert_all_id=True)
    H_mpo = graph.build_MPO()

    return H_mpo, lat


if __name__ == '__main__':
    L = 60

    # 1. fit 1/r with exponential
    N_max = 100
    K = 8

    lambdas, alphas, n, y, y_fit = fit_inverse_with_exponentials(
        N_max=N_max,
        K=K,
        mu_min=1e-3,
        mu_max=1.0,
        weight_power=1.0,
    )

    # 2. use the fit result produce Coulomb MPO
    H_mpo, lat = make_coulomb_mpo_from_fit(L, alphas, lambdas, bc='finite')

    # 3. construct a half filled product state MPS
    sites = lat.mps_sites()
    print("local basis states:", sites[0].state_labels)
    empty, full = sites[0].state_labels

    # first half filled, second half empty
    N_full = L // 2
    # init_state = [full] * (N_full) + [empty] * (L - N_full)
    init_state = [empty] * L

    # randomly choose which sites are full
    full_sites = random.sample(range(L), N_full)

    for i in full_sites:
        init_state[i] = full

    psi = MPS.from_product_state(sites, init_state, 'finite')

    # 4. print the MPS
    n_expect = psi.expectation_value('N')
    print("⟨n_i⟩ =", n_expect)
    print("total filling =", np.sum(n_expect), "/", L)

    # 5. MPO bond dimension
    print("MPO bond dimensions per site:", H_mpo.chi)

    # MPO expectation energy
    E = H_mpo.expectation_value(psi)
    print("⟨H⟩ on this half-filled product state =", E)

    # 6. DMRG
    model = MPOModel(lat, H_mpo)

    dmrg_params = {
        'mixer': True,  # 帮助跳出local minima
        'mixer_params': {
            'amplitude': 1e-3,
            'decay': 1.01,
        },
        'trunc_params': {
            'chi_min': 5, 
            'chi_max': 100,   # 不够再往上加
            'svd_min': 1e-9,
        },
        'max_sweeps': 1000,
        'max_E_err': 1e-8,
    }

    info = dmrg.run(psi, model, dmrg_params)
    E0 = info['E'] 

    print("DMRG ground state energy =", E0)
    n_expect = psi.expectation_value('N')
    print("⟨n_i⟩ =", n_expect)
    print(psi.chi)