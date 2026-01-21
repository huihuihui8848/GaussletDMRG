import numpy as np
import matplotlib.pyplot as plt

# ========= 你的已给函数（未归一化 Gaussian） =========
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

# ---------- Analytic derivatives ----------
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

# ========= 辅助：把 b_{-j}=b_{+j}=b[j] 展开成全索引 =========
def _full_indexed_coeffs(b):
    b = np.asarray(b, dtype=float)
    J = len(b) - 1
    j_pos = np.arange(0, J+1)
    j_full = np.concatenate((-j_pos[:0:-1], j_pos))   # [-J..-1,0,1..J]
    c_full = b[np.abs(j_full)]
    return j_full.astype(float), c_full.astype(float)

# ========= 闭式 S_k, T_k（自动匹配常数） =========
def Sk_closed(b, k, use_normalized_constants=False):
    j_full, c_full = _full_indexed_coeffs(b)
    J = j_full[:, None]
    JP = j_full[None, :]
    coeffs = (c_full[:, None] * c_full[None, :])
    delta = 3.0 * float(k) + (J - JP)
    core = np.sum(coeffs * np.exp(-(delta**2) / 4.0))
    if use_normalized_constants:
        return (1.0 / (6.0 * np.sqrt(np.pi))) * core
    else:
        return (np.sqrt(np.pi) / 3.0) * core

def Tk_closed(b, k, s, use_normalized_constants=False):
    j_full, c_full = _full_indexed_coeffs(b)
    J = j_full[:, None]
    JP = j_full[None, :]
    coeffs = (c_full[:, None] * c_full[None, :])
    delta = 3.0 * float(k) + (J - JP)
    base = np.exp(-(delta**2) / 4.0)
    bracket = 0.5 - (delta**2) / 4.0  # 采用 T=∫(G')(G')dx 的正定表达
    core = np.sum(coeffs * base * bracket)
    if use_normalized_constants:
        pref = 3.0 / (2.0 * np.sqrt(np.pi) * s**2)
    else:
        pref = 3.0 * np.sqrt(np.pi) / (s**2)
    return pref * core

# ========= 数值积分 S_k, T_k =========
def Sk_numeric(b, k, s=1.0, window_mult=12, dx=None):
    n, nprime = 0, int(k)
    c0, c1 = n * s, nprime * s
    L0, L1 = c0 - window_mult * s, c1 + window_mult * s
    if dx is None:
        dx = s / 40.0
    x = np.arange(L0, L1 + dx, dx)
    G0 = gausslet_scaled(x, b, center=c0, s=s)
    G1 = gausslet_scaled(x, b, center=c1, s=s)
    return np.trapz(G0 * G1, x)

def Tk_numeric(b, k, s=1.0, window_mult=12, dx=None):
    n, nprime = 0, int(k)
    c0, c1 = n * s, nprime * s
    L0, L1 = c0 - window_mult * s, c1 + window_mult * s
    if dx is None:
        dx = s / 40.0
    x = np.arange(L0, L1 + dx, dx)
    G0p = gausslet_scaled_prime(x, b, center=c0, s=s)
    G1p = gausslet_scaled_prime(x, b, center=c1, s=s)
    return np.trapz(G0p * G1p, x)

# ========= 读入 b_j =========
_, b = np.loadtxt('coefficient/G4.csv', unpack=True, delimiter=',')

# ========= 公共参数 =========
use_normalized_constants = False
eps = 1e-16
KMAX = 25
ks = np.arange(0, KMAX + 1, dtype=int)

# ========= 画 S_k（原逻辑不变） =========
S_num = np.array([Sk_numeric(b, k, s=1.0) for k in ks])
S_clo = np.array([Sk_closed(b, k, use_normalized_constants=use_normalized_constants) for k in ks])

plt.figure()
plt.semilogy(ks, np.abs(S_clo), label="S_k (closed)")
plt.semilogy(ks, np.abs(S_num), marker="o", linestyle="none", label="S_k (numeric)")
plt.hlines(eps, 0, KMAX+1, colors='g', label='epsilon=1e-16')
plt.title("Overlap S_k")
plt.xlabel("k = |n - n'|")
plt.ylabel("|S_k|")
plt.legend()
plt.savefig('img/overlap.png')
plt.show()

# ========= 新增：多个 s 的 T_k 同图对比 =========
s_list = [1, 10, 0.1, 0.01]   # <<< 你要比较的尺度放这里

plt.figure()
for s in s_list:
    T_clo_s = np.array([Tk_closed(b, k, s=s, use_normalized_constants=use_normalized_constants) for k in ks])
    T_num_s = np.array([Tk_numeric(b, k, s=s) for k in ks])
    # 闭式曲线（线），数值点（散点）：
    plt.semilogy(ks, np.abs(T_clo_s) + 1e-300, label=f"|T_k| closed, s={s:g}")
    plt.semilogy(ks, np.abs(T_num_s) + 1e-300, linestyle="none", marker="o", label=f"|T_k| numeric, s={s:g}")

plt.hlines(eps, 0, KMAX+1, colors='g', label='epsilon=1e-16')
plt.title("Kinetic |T_k|")
plt.xlabel("k = |n - n'|")
plt.ylabel("|T_k|")
plt.legend(ncol=2)
plt.savefig('img/kinetic.png')
plt.show()
