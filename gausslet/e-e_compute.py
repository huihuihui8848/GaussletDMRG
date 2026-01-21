import numpy as np

####################################
# Gausslet Def.
####################################

def gaussian(x, mu=0.0, sigma=1.0):
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = float(sigma)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def gausslet_primary(x, b):
    """
    最原始的 gausslet 组合:
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
    缩放 + 平移后的基函数:
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
# 构造 basis 在网格上的取值
####################################

def build_basis_values(x, b, centers, scales):
    """
    在给定网格 x 上计算一组 gausslet basis:
        phi_i(x) = gausslet_scaled(x, b, center_i, scale_i)
    参数:
        x       : 1D numpy array, 实空间网格
        b       : gausslet_primary 的系数数组
        centers : 每个基函数的中心位置数组, 长度 = n_basis
        scales  : 每个基函数的尺度 s_i 数组, 长度 = n_basis
    返回:
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
# 计算四指标 Coulomb 张量 V_ijkl
####################################

def build_coulomb_tensor(b, centers, scales,
                         L=10.0, n_grid=400,
                         softening=1e-2):
    """
    用 gausslet basis 数值计算 1D Coulomb 积分:
        V_ijkl = ∫∫ dx dx' phi_i(x) phi_j(x)
                 (1 / sqrt((x - x')^2 + softening^2))
                 phi_k(x') phi_l(x')

    参数:
        b        : gausslet_primary 的系数数组
        centers  : 每个基函数的中心 (array, 长度 = n_basis)
        scales   : 每个基函数的尺度 (array, 长度 = n_basis)
        L        : 实空间积分区间 [-L, L]
        n_grid   : 网格点数
        softening: kernel 中的软化参数 (避免 x=x' 发散)

    返回:
        x : 实空间网格, shape = (Nx,)
        V : Coulomb 张量, shape = (n_basis, n_basis, n_basis, n_basis)
    """
    # 1. 实空间网格
    x = np.linspace(-L, L, n_grid)
    dx = x[1] - x[0]

    # 2. basis 在网格上的取值 Phi[i,a] = phi_i(x[a])
    Phi = build_basis_values(x, b, centers, scales)
    print(Phi.shape)
    n_basis = Phi.shape[0]

    # 3. 构造 Coulomb kernel K[a,b]
    K = coulomb_kernel_1d(x, x, softening=softening)  # shape (Nx, Nx)
    print(K)

    # 4. 预计算 phi_i(x)phi_j(x) 和 phi_k(x')phi_l(x')
    #    P[i,j,a] = phi_i(a) * phi_j(a)
    P = np.einsum('ia,ja->ija', Phi, Phi)   # (n_basis, n_basis, Nx)
    #    Q[k,l,b] = phi_k(b) * phi_l(b)
    #    在这里 Q 和 P 在数值结构上是一样的，只是索引名字不同，
    #    为了易读性直接复制一份
    Q = P.copy()  # (n_basis, n_basis, Nx)

    # 5. 用 einsum 做双重求和：
    #    V_ijkl = sum_{a,b} P[i,j,a] * K[a,b] * Q[k,l,b] * dx^2
    V = np.einsum('ija,ab,klb->ijkl', P, K, Q) * dx * dx

    return x, V

def make_centers_and_scales(L, N):
    """
    生成:
        centers: [0, -2s, 2s, -4s, 4s, ...] 对称排布
        scales : 每个基函数同一个尺度 s = L / N

    要求:
        N = 1 + 2*M 为奇数 (一个在 0, 其余成 ± 成对)
    """
    if N % 2 == 0:
        raise ValueError(
            f"N = {N} 是偶数。要 0, ±2s, ±4s,... 这种对称排布时，"
            "基函数数目 N 必须是奇数 (N = 1 + 2*M)。"
        )

    s = L / N                    # 你原来写的 scales = L/N
    centers = np.zeros(N, dtype=float)

    M = (N - 1) // 2             # 一共有 M 对 ±
    for i in range(1, M + 1):
        centers[2 * i - 1] = -2 * i * s   # 奇数下标: 负
        centers[2 * i]     =  2 * i * s   # 偶数下标: 正

    scales = np.full(N, s, dtype=float)   # 每个基函数同一个 scale
    return centers, scales



if __name__ == "__main__":
    _, b = np.loadtxt('coefficient/G4.csv', unpack=True, delimiter=',')
    L = 10.0
    N = 31
    centers, scales = make_centers_and_scales(L, N)


    # 计算 Coulomb 张量
    x, V = build_coulomb_tensor(b, centers, scales,
                                L=L, n_grid=500,
                                softening=1e-2)

    # 举例看一个元件:
    print(x)
    print(V)
    np.savez('data/V', V=V)
