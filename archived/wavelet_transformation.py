from typing import Callable, Iterable, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed

ArrayLike = Iterable[float]

def cwt_transform(
    f: Callable[[np.ndarray], np.ndarray] | object,
    psi: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    scales: ArrayLike,
    positions: ArrayLike,
    *,
    n_samples: int = 4096,
    chunk_size: Optional[int] = None,   # 防止内存爆：按 b 分块
    dtype = np.complex128,
) -> np.ndarray:
    """
    连续小波变换 W(a,b) 的数值实现；对每个尺度 a，向量化计算所有 b。

    参数
    ----
    f          : 可向量化的信号函数，或实现了 __call__(x)->array 的对象（如你的 Function 子类）
    psi        : 母小波 ψ(u)，须支持数组 u（可 2D）
    x_min/max  : 积分区间
    scales     : a 网格（>0）
    positions  : b 网格
    n_samples  : x 方向积分采样数（均匀）
    chunk_size : 按 b 分块的大小（None=不分块；建议当 |b|×n_samples 太大时设为几百~几千）
    dtype      : 计算用复数类型

    返回
    ----
    W : (len(scales), len(positions)) 的复数组
    """
    # --- 网格 & 取样 ---
    x = np.linspace(x_min, x_max, int(n_samples), endpoint=True)
    dx = (x_max - x_min) / (int(n_samples) - 1)

    # 允许传入 Function 子类实例（实现了 __call__），或普通 callable
    fx = np.asarray(f(x))  # 若 f 是 Function 子类，会走到 Function.__call__ -> evaluate
    if fx.ndim != 1:
        raise ValueError("f(x) 必须返回 1D 数组。")
    fx = fx.astype(dtype, copy=False)

    scales = np.asarray(scales, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if np.any(scales <= 0):
        raise ValueError("所有 scale a 必须为正。")

    A = scales.size
    B = positions.size
    W = np.zeros((A, B), dtype=dtype)

    # --- b 分块，防止一次性分配 (B, N) 太大 ---
    if chunk_size is None or chunk_size <= 0:
        chunk_size = B

    for ia, a in enumerate(scales):
        norm = 1.0 / np.sqrt(a)
        # 分块遍历 b
        for start in range(0, B, chunk_size):
            stop = min(B, start + chunk_size)
            b_chunk = positions[start:stop]              # (K,)
            # 2D U: (K, N)，一次性算出该尺度下 K 个 b 的 ψ 值
            U = (x[None, :] - b_chunk[:, None]) / a      # (K, N)
            psi_vals = psi(U)                             # (K, N)（复/实均可）
            integrand = fx[None, :] * np.conjugate(psi_vals) * norm  # (K, N)
            # 沿 x 积分，得到 (K,)
            W[ia, start:stop] = np.trapz(integrand, dx=dx, axis=1)

    return W


def cwt_transform_parallel(
    f: Callable[[np.ndarray], np.ndarray] | object,
    psi: Callable[[np.ndarray], np.ndarray],
    x_min: float,
    x_max: float,
    scales: ArrayLike,
    positions: ArrayLike,
    *,
    n_samples: int = 4096,
    chunk_size: Optional[int] = None,
    n_workers: int = 4,
    dtype = np.complex128,
) -> np.ndarray:
    """
    并行计算 W(a,b)，每个尺度 a 由一个线程负责；在该线程内部对 b 进行向量化/分块。
    """
    # --- 网格 & 取样 ---
    x = np.linspace(x_min, x_max, int(n_samples), endpoint=True)
    dx = (x_max - x_min) / (int(n_samples) - 1)

    fx = np.asarray(f(x))  # 支持 Function 子类（走 __call__）
    if fx.ndim != 1:
        raise ValueError("f(x) 必须返回 1D 数组。")
    fx = fx.astype(dtype, copy=False)

    a = np.asarray(scales, dtype=float)
    b = np.asarray(positions, dtype=float)
    if np.any(a <= 0):
        raise ValueError("所有 scale a 必须为正。")

    A, B = a.size, b.size
    W = np.zeros((A, B), dtype=dtype)

    if chunk_size is None or chunk_size <= 0:
        chunk_size = B

    # 单个尺度的工作函数（在线程内运行）
    def _work_one_scale(ia: int) -> tuple[int, np.ndarray]:
        ai = a[ia]
        norm = 1.0 / np.sqrt(ai)
        Wi = np.zeros(B, dtype=dtype)
        # 分块迭代 b
        for start in range(0, B, chunk_size):
            stop = min(B, start + chunk_size)
            b_chunk = b[start:stop]                       # (K,)
            U = (x[None, :] - b_chunk[:, None]) / ai      # (K, N)
            psi_vals = psi(U)                              # (K, N)
            integrand = fx[None, :] * np.conjugate(psi_vals) * norm  # (K, N)
            Wi[start:stop] = np.trapz(integrand, dx=dx, axis=1)      # (K,)
        return ia, Wi

    # 线程池并行各尺度
    workers = max(1, int(n_workers))
    if workers == 1:
        # 退化为串行
        for ia in range(A):
            _, Wi = _work_one_scale(ia)
            W[ia, :] = Wi
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_work_one_scale, ia) for ia in range(A)]
            for fut in as_completed(futures):
                ia, Wi = fut.result()
                W[ia, :] = Wi

    return W


def plot_cwt(
    W: np.ndarray,
    scales: ArrayLike,
    positions: ArrayLike,
    *,
    title: str | None = None,
    yscale: str | None = None,  # 如需对 a 轴用 'log'
) -> None:
    """画 |W| 的 2D 热力图；纵轴为 a，小在上大在下。"""
    scales = np.asarray(scales, dtype=float)
    positions = np.asarray(positions, dtype=float)
    magnitude = np.abs(W)

    plt.figure(figsize=(8, 4.8))
    extent = [positions.min(), positions.max(), scales.max(), scales.min()]
    plt.imshow(magnitude, aspect='auto', extent=extent, origin='upper')
    plt.xlabel("Position b")
    plt.ylabel("Scale a")
    plt.title(title if title else "CWT |W(a,b)|")
    plt.colorbar(label="|W|")
    if yscale == "log":
        # 若想用 log a 显示，可改为 pcolormesh + 对网格做 log 变换
        pass
    plt.tight_layout()
    plt.show()


def save_cwt(
    path: str,
    W: np.ndarray,
    scales: ArrayLike,
    positions: ArrayLike,
    *,
    meta: Optional[Dict[str, Any]] = None,
    compress: bool = True,
) -> None:
    """
    保存 CWT 结果为 .npz；可携带元数据（domain、n_samples、wavelet_tag 等）。
    """
    data = {
        "W": np.asarray(W),
        "scales": np.asarray(scales, dtype=float),
        "positions": np.asarray(positions, dtype=float),
    }
    if meta is not None:
        # 将简单标量/小数组元信息并入；复杂对象建议用户自行管理
        for k, v in meta.items():
            # 仅保存可序列化为 np.array 或标量/字符串的值
            try:
                data[k] = np.array(v)
            except Exception:
                pass
    if compress:
        np.savez_compressed(path, **data)
    else:
        np.savez(path, **data)


def load_cwt(path: str) -> Dict[str, Any]:
    """
    读取 .npz 文件，返回 dict：{"W", "scales", "positions", 以及保存时附带的 meta 字段}
    """
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    return out



def morlet_wavelet(w0: float = 6.0) -> Callable[[np.ndarray], np.ndarray]:
    norm = np.pi ** (-0.25)
    def psi(u: np.ndarray) -> np.ndarray:
        return norm * np.exp(1j * w0 * u) * np.exp(-0.5 * (u ** 2))
    return psi

def mexican_hat() -> Callable[[np.ndarray], np.ndarray]:
    coef = 2.0 / (np.sqrt(3.0) * (np.pi ** 0.25))
    def psi(u: np.ndarray) -> np.ndarray:
        return coef * (1.0 - u**2) * np.exp(-0.5 * u**2)
    return psi





if __name__ == "__main__":
    pass
