
"""
adaptive_gausslet.py

Route A: 自适应（一维）gausslet 逼近/网格细化
--------------------------------------------
思路：在 dyadic 层级 j=J_min..J_max 上，用尺度 s_j = s0 / 2^j 的 gausslet。
从粗到细逐层细化：在当前激活的区间上拟合 -> 计算残差能量 -> 超过阈值就细化该区间，
在下一层加入该区间的子区间（对应的 gausslet 中心），并重新拟合，直至收敛。

依赖：
- gausslet.gausslet_scaled : 生成缩放+平移的 gausslet
- 可选：用户可替换拟合器；默认使用本文件的加权最小二乘解算器（稳定性可加 Tikhonov 正则）

API 概览：
- AdaptiveGaussletFitter(...): 配置器/求解器
- fit(f): 运行自适应细化循环，返回结果对象 FitResult
- result.evaluate(x): 在采样点上重构
- result.summary(): 打印层级/系数统计
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    # Prefer package-style imports if placed inside a package
    from .gausslet import gausslet_scaled
except Exception:
    # Fallback to local import path when run as a script next to gausslet.py
    from gausslet import gausslet_scaled  # type: ignore


# ---------------------- 工具函数 ----------------------

def _trapezoidal_weights(xs: np.ndarray) -> np.ndarray:
    if xs.ndim != 1 or xs.size < 2:
        raise ValueError("quadrature grid must be 1D with at least two points")
    diffs = np.diff(xs)
    if np.any(diffs <= 0.0):
        raise ValueError("quadrature grid must be strictly increasing")
    w = np.empty_like(xs, dtype=float)
    w[0] = 0.5 * diffs[0]
    w[-1] = 0.5 * diffs[-1]
    if xs.size > 2:
        w[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return w


def _coverage_margin(bcoef: Sequence[float], s: float, pad_sigma: float) -> float:
    """
    基于 gausslet 内部子高斯的最远中心与 ~pad_sigma*sigma_x 计算安全边界。
    """
    bcoef = np.asarray(bcoef, dtype=float)
    J = len(bcoef) - 1
    sigma_x = s / 3.0
    far_internal = (J / 3.0) * s
    return float(far_internal + pad_sigma * sigma_x)


def _make_quadrature_grid(interval: Tuple[float, float],
                          bcoef: Sequence[float],
                          s: float,
                          dx: float | None,
                          pad_sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    left, right = float(interval[0]), float(interval[1])
    if dx is None:
        dx = s / 200.0
    margin = _coverage_margin(bcoef, s, pad_sigma)
    xs = np.arange(left - margin, right + margin + dx, dx, dtype=float)
    w = _trapezoidal_weights(xs)
    return xs, w


def _centers_cover_interval(interval: Tuple[float, float],
                            bcoef: Sequence[float],
                            s: float,
                            pad_sigma: float) -> np.ndarray:
    """
    在步长 s 的均匀网格上，返回完全覆盖 interval 的中心集合。
    """
    left, right = float(interval[0]), float(interval[1])
    margin = _coverage_margin(bcoef, s, pad_sigma)
    k0 = int(np.floor((left - margin) / s))
    k1 = int(np.ceil((right + margin) / s))
    return s * np.arange(k0, k1 + 1, dtype=float)


# ---------------------- 数据结构 ----------------------

@dataclass
class BasisAtom:
    """单个基函数（节点）"""
    j: int              # 层级
    c: float            # 中心
    s: float            # 尺度 s_j = s0 / 2^j


@dataclass
class FitResult:
    atoms: List[BasisAtom]                 # 激活基函数列表（有序）
    coeffs: np.ndarray                     # shape = (M,)
    xs: np.ndarray                         # 拟合时使用的积分格点
    w: np.ndarray                          # 积分权重（梯形）
    residual_rms: float                    # 全局 RMS 残差
    per_level_counts: Dict[int, int]       # 各层基函数数量

    def evaluate(self, x: Sequence[float] | np.ndarray,
                 bcoef: Sequence[float]) -> np.ndarray:
        """在 x 上重构函数值"""
        x = np.asarray(x, dtype=float)
        cols = [gausslet_scaled(x, bcoef, center=a.c, s=a.s) for a in self.atoms]
        if len(cols) == 0:
            return np.zeros_like(x)
        Phi = np.stack(cols, axis=1)  # (N, M)
        return Phi @ self.coeffs

    def summary(self) -> str:
        levels = sorted(self.per_level_counts.keys())
        parts = [f"total atoms: {len(self.atoms)}, residual RMS: {self.residual_rms:.3e}"]
        for j in levels:
            parts.append(f"  level j={j}: {self.per_level_counts[j]} atoms")
        return "\n".join(parts)


# ---------------------- 主类：自适应拟合器 ----------------------

class AdaptiveGaussletFitter:
    """
    自适应 gausslet 拟合（1D）。

    Parameters
    ----------
    bcoef : Sequence[float]
        gausslet 主函数系数（与 gausslet_scaled 一致用法）。
    interval : (left, right)
        目标函数的物理区间。
    s0 : float
        最粗层尺度。
    J_min, J_max : int
        自适应层级范围（包含）。
    dx : float | None
        积分网格步长。若为 None，自动取 s_j/200（每层会取当前最细尺度）。
    pad_sigma : float
        拓展区间的 sigma 余量，默认 6。
    reg : float
        Tikhonov 正则（对法方程加 reg*I），提高病态时的稳定性。
    refine_factor : float
        细化阈值（相对全局 RMS）。若父区间 RMS > refine_factor * 全局 RMS，则细化。
    max_iters : int
        最大自适应迭代次数。
    """

    def __init__(self,
                 bcoef: Sequence[float],
                 interval: Tuple[float, float] = (-4.0, 4.0),
                 s0: float = 1.0,
                 J_min: int = 0,
                 J_max: int = 3,
                 dx: float | None = None,
                 pad_sigma: float = 6.0,
                 reg: float = 0.0,
                 refine_factor: float = 0.8,
                 max_iters: int = 6):
        self.bcoef = np.asarray(bcoef, dtype=float)
        self.interval = (float(interval[0]), float(interval[1]))
        self.s0 = float(s0)
        self.J_min = int(J_min)
        self.J_max = int(J_max)
        if self.J_max < self.J_min:
            raise ValueError("J_max must be >= J_min")
        self.dx = dx
        self.pad_sigma = float(pad_sigma)
        self.reg = float(reg)
        self.refine_factor = float(refine_factor)
        self.max_iters = int(max_iters)

    # ---- 构建/维护基 ----

    def _build_design(self, atoms: List[BasisAtom], xs: np.ndarray) -> np.ndarray:
        """构建设计矩阵 B（N, M），列为各基在 xs 上的取值。"""
        cols = [gausslet_scaled(xs, self.bcoef, center=a.c, s=a.s) for a in atoms]
        if len(cols) == 0:
            return np.zeros((xs.size, 0), dtype=float)
        return np.stack(cols, axis=1)

    def _solve_weighted_ls(self, B: np.ndarray, y: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, float]:
        """解 (B^T W B) x = B^T W y，返回系数与条件数。"""
        W = w[:, None]
        S = B.T @ (B * W)          # 法方程矩阵
        rhs = B.T @ (y * w)        # 右端项
        if self.reg > 0.0:
            S = S + self.reg * np.eye(S.shape[0], dtype=S.dtype)
        try:
            coeffs = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            coeffs, *_ = np.linalg.lstsq(S, rhs, rcond=None)
        try:
            cond_S = float(np.linalg.cond(S))
        except np.linalg.LinAlgError:
            cond_S = float(np.inf)
        return coeffs, cond_S

    def _initial_atoms(self) -> List[BasisAtom]:
        """在最粗层用步长 s0 覆盖区间。"""
        s = self.s0 / (2 ** self.J_min)
        centers = _centers_cover_interval(self.interval, self.bcoef, s, self.pad_sigma)
        return [BasisAtom(self.J_min, float(c), s) for c in centers]

    # ---- 自适应循环 ----

    def fit(self, f: Callable[[np.ndarray], np.ndarray]) -> FitResult:
        # 以当前最细尺度设置积分网格（稳定起见，取 J_max 的 s）
        s_fine = self.s0 / (2 ** self.J_max)
        xs, w = _make_quadrature_grid(self.interval, self.bcoef, s_fine, self.dx, self.pad_sigma)
        y = np.asarray(f(xs), dtype=float)
        if y.shape != xs.shape:
            raise ValueError("f(xs) must return shape equal to xs")

        # 初始：最粗层覆盖
        atoms: List[BasisAtom] = self._initial_atoms()

        for _ in range(self.max_iters):
            # 拟合
            B = self._build_design(atoms, xs)
            coeffs, _cond = self._solve_weighted_ls(B, y, w)
            y_rec = B @ coeffs
            r = y - y_rec
            # 全局 RMS
            w_sum = float(np.sum(w))
            global_rms = float(np.sqrt(np.sum(w * r * r) / w_sum))

            # 终止条件：无可细化或已达最细层
            if global_rms == 0.0:
                break

            # 逐层检查父区间的残差 RMS，决定是否细化（添加子区间的基）
            # 父区间定义：长度等于 s_j，中心为当前层的网格点
            refined_any = False
            new_atoms: List[BasisAtom] = atoms.copy()
            # 记录已有的 (j,c) -> True，防止重复添加
            existing = {(a.j, a.c): True for a in atoms}

            # 在 j=J_min..J_max-1 上尝试细化
            for j in range(self.J_min, self.J_max):
                s_j = self.s0 / (2 ** j)
                s_child = s_j / 2.0
                # 以步长 s_j 的网格中心，枚举父区间
                centers_j = _centers_cover_interval(self.interval, self.bcoef, s_j, self.pad_sigma)
                # 每个父区间 I = [c - s_j/2, c + s_j/2]
                half = 0.5 * s_j
                for c in centers_j:
                    left = c - half
                    right = c + half
                    # 计算该区间内的残差 RMS
                    mask = (xs >= left) & (xs < right)
                    if not np.any(mask):
                        continue
                    w_loc = w[mask]
                    r_loc = r[mask]
                    rms_loc = float(np.sqrt(np.sum(w_loc * r_loc * r_loc) / np.sum(w_loc)))
                    if rms_loc > self.refine_factor * global_rms:
                        # 细化：添加两个子区间的中心（c - s_child/2, c + s_child/2）
                        c_left = c - 0.5 * s_child
                        c_right = c + 0.5 * s_child
                        # 仅当下一层未超过 J_max 且尚未存在时添加
                        keyL = (j + 1, float(c_left))
                        keyR = (j + 1, float(c_right))
                        if (j + 1) <= self.J_max:
                            if keyL not in existing:
                                new_atoms.append(BasisAtom(j + 1, float(c_left), s_child))
                                existing[keyL] = True
                                refined_any = True
                            if keyR not in existing:
                                new_atoms.append(BasisAtom(j + 1, float(c_right), s_child))
                                existing[keyR] = True
                                refined_any = True

            atoms = new_atoms
            if not refined_any:
                # 没有任何细化，终止
                break

        # 最终求解一次，生成结果
        B = self._build_design(atoms, xs)
        coeffs, _ = self._solve_weighted_ls(B, y, w)
        y_rec = B @ coeffs
        r = y - y_rec
        residual_rms = float(np.sqrt(np.sum(w * r * r) / np.sum(w)))

        # 统计每层基函数数量
        per_level: Dict[int, int] = {}
        for a in atoms:
            per_level[a.j] = per_level.get(a.j, 0) + 1

        return FitResult(atoms=atoms, coeffs=coeffs, xs=xs, w=w,
                         residual_rms=residual_rms, per_level_counts=per_level)


# ---------------------- 简易演示 ----------------------

def _demo():
    # 生成一组示例 b 系数（可替换为你自己的）
    # 这里用一个快速收敛的衰减序列作为占位
    coe = "coefficient/G4.csv"

    # Robustly load coefficients: accept 1-col (b) or 2-col (j,b)
    arr = np.loadtxt(coe, delimiter=",")
    if arr.ndim == 1:   # single column
        b = arr.astype(float)
    else:
        # assume first column is j (0..M-1), second column is b
        # sort by j just in case
        order = np.argsort(arr[:, 0])
        j_in = arr[order, 0].astype(float)
        b = arr[order, 1].astype(float)
        # optional sanity: j_in should be 0..M-1 (within tol); otherwise we still proceed
        # but you can assert if you require exact integer grid:
        # assert np.allclose(j_in, np.arange(len(b), dtype=float))

    # 目标函数：分段平缓 + 局部尖峰
    def f(x: np.ndarray) -> np.ndarray:
        return (
            0.3 * np.sin(0.6 * x)
            + np.exp(-0.5 * ((x - 1.5) / 0.12) ** 2)
            + 0.4 * np.exp(-0.5 * ((x + 2.0) / 0.25) ** 2)
        )

    fitter = AdaptiveGaussletFitter(
        bcoef=b, interval=(-4.0, 4.0), s0=1.0,
        J_min=0, J_max=4, dx=None, pad_sigma=6.0,
        reg=1e-10, refine_factor=0.9, max_iters=6
    )
    res = fitter.fit(f)

    # 打印摘要
    print(res.summary())

    # 简单可视化（需要 matplotlib）

    import matplotlib.pyplot as plt
    xs = res.xs
    xx = np.linspace(-8, 8, 100)
    plt.plot(xx, f(xx), label="f(x)")
    plt.plot(xs, res.evaluate(xs, b), label="reconstruction")
    # 标出不同层的中心
    for j, cnt in res.per_level_counts.items():
        cs = [a.c for a in res.atoms if a.j == j]
        ys = res.evaluate(np.array(cs), b)
        plt.scatter(cs, ys, s=12, label=f"centers j={j}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    _demo()
