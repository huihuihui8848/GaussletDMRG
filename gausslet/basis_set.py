from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = ["basis_set"]


def _gaussian(x: np.ndarray | float, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    sigma_val = float(sigma)
    return np.exp(-0.5 * ((x_arr - mu_arr) / sigma_val) ** 2)


def _gausslet_primary(x: np.ndarray | Sequence[float], bcoef: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    x3 = 3.0 * x_arr
    j = np.arange(bcoef.size, dtype=float)

    if j.shape != bcoef.shape:
        raise ValueError("bcoef must define coefficients for j = 0..M-1")

    result = bcoef[0] * _gaussian(x3, j[0], sigma=1.0)
    if j.size > 1:
        g_pos = _gaussian(x3[:, None], j[1:], sigma=1.0)
        g_neg = _gaussian(x3[:, None], -j[1:], sigma=1.0)
        result += g_pos @ bcoef[1:] + g_neg @ bcoef[1:]
    return result


def _gausslet_scaled(
    x: np.ndarray | Sequence[float], bcoef: np.ndarray, center: float, scale: float
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    inv_root_scale = 1.0 / np.sqrt(scale)
    return inv_root_scale * _gausslet_primary((x_arr - center) / scale, bcoef)


def _trapezoidal_weights(xs: np.ndarray) -> np.ndarray:
    if xs.ndim != 1 or xs.size < 2:
        raise ValueError("quadrature grid must be one-dimensional with at least two points")

    diffs = np.diff(xs)
    if np.any(diffs <= 0.0):
        raise ValueError("quadrature grid must be strictly increasing")

    weights = np.empty_like(xs, dtype=float)
    weights[0] = 0.5 * diffs[0]
    weights[-1] = 0.5 * diffs[-1]
    if xs.size > 2:
        weights[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return weights


def _overlap_matrix(
    functions: Sequence[Callable[[np.ndarray], np.ndarray]],
    domain: np.ndarray | Sequence[float],
    weights: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    funcs = list(functions)
    if not funcs:
        return np.zeros((0, 0), dtype=float)

    xs = np.asarray(domain, dtype=float)
    if xs.ndim != 1:
        raise ValueError("domain must be one-dimensional")

    if weights is None:
        w = _trapezoidal_weights(xs)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != xs.shape:
            raise ValueError("weights must match domain shape")

    columns: list[np.ndarray] = []
    for func in funcs:
        values = np.asarray(func(xs), dtype=float)
        if values.shape != xs.shape:
            raise ValueError("each function must return values matching domain shape")
        columns.append(values)

    B = np.stack(columns, axis=1)
    return B.T @ (B * w[:, None])


class basis_set:
    """Collection of gausslet basis functions built on a shared coefficient table."""

    def __init__(
        self,
        bcoef: Sequence[float],
        centers: Sequence[float],
        scales: float | Sequence[float] | None = None,
        weights: Sequence[float] | None = None,
        domain: tuple[float, float] = (-4.0, 4.0),
    ) -> None:
        self.bcoef = np.asarray(bcoef, dtype=float)
        if self.bcoef.ndim != 1 or self.bcoef.size == 0:
            raise ValueError("bcoef must be a non-empty 1D sequence")

        self.centers = np.asarray(list(centers), dtype=float)
        if self.centers.ndim != 1 or self.centers.size == 0:
            raise ValueError("centers must be a non-empty 1D sequence")

        if scales is None:
            self.scales = np.full_like(self.centers, fill_value=1.0, dtype=float)
        else:
            if np.isscalar(scales):
                self.scales = np.full_like(self.centers, fill_value=float(scales), dtype=float)
            else:
                self.scales = np.asarray(list(scales), dtype=float)
            if self.scales.shape != self.centers.shape:
                raise ValueError("scales must match centers length")
        if np.any(self.scales <= 0.0):
            raise ValueError("scales must be positive")

        if weights is None:
            self.weights = np.ones_like(self.centers, dtype=float)
        else:
            self.weights = np.asarray(list(weights), dtype=float)
            if self.weights.shape != self.centers.shape:
                raise ValueError("weights must match centers length")

        domain = (float(domain[0]), float(domain[1]))
        if not domain[0] < domain[1]:
            raise ValueError("domain must satisfy left < right")
        self.domain = domain

        self.basis_functions = [
            self._make_basis(c, s, w) for c, s, w in zip(self.centers, self.scales, self.weights)
        ]

    def _make_basis(self, center: float, scale: float, weight: float) -> Callable[[np.ndarray], np.ndarray]:
        def basis(x: np.ndarray) -> np.ndarray:
            values = _gausslet_scaled(x, self.bcoef, center=center, scale=scale)
            return weight * values

        return basis

    def _support_radius(self) -> float:
        J = self.bcoef.size - 1
        far_internal = (J / 3.0) * np.max(self.scales)
        sigma_x = np.max(self.scales) / 3.0
        tail = 6.0 * sigma_x
        return far_internal + tail

    def _quadrature_grid(
        self, dx: float | None = None, num_points: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if dx is not None and num_points is not None:
            raise ValueError("specify at most one of dx or num_points")

        support = self._support_radius()
        left = min(self.domain[0], float(np.min(self.centers) - support))
        right = max(self.domain[1], float(np.max(self.centers) + support))

        if num_points is not None:
            if num_points < 2:
                raise ValueError("num_points must be at least 2")
            xs = np.linspace(left, right, int(num_points), dtype=float)
            weights = _trapezoidal_weights(xs)
            return xs, weights

        if dx is None:
            base = float(np.min(self.scales))
            dx = base / 200.0

        xs = np.arange(left, right + dx, dx, dtype=float)
        weights = _trapezoidal_weights(xs)
        return xs, weights

    def fit(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        dx: float | None = None,
        num_points: int | None = None,
        regularization: float = 0.0,
    ) -> np.ndarray:
        xs, weights = self._quadrature_grid(dx=dx, num_points=num_points)
        columns = [basis(xs) for basis in self.basis_functions]
        design = np.stack(columns, axis=1)

        f_vals = np.asarray(f(xs), dtype=float)
        if f_vals.shape != xs.shape:
            raise ValueError("f(xs) must match quadrature grid shape")

        w_col = weights[:, None]
        gram = design.T @ (design * w_col)
        rhs = design.T @ (f_vals * weights)

        if regularization > 0.0:
            gram = gram + regularization * np.eye(gram.shape[0], dtype=gram.dtype)

        try:
            coeffs = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            coeffs, *_ = np.linalg.lstsq(gram, rhs, rcond=None)

        self._last_fit = {
            "coeffs": coeffs,
            "grid": xs,
            "weights": weights,
            "design": design,
        }
        return coeffs

    def overlap_matrix(
        self, dx: float | None = None, num_points: int | None = None
    ) -> np.ndarray:
        xs, weights = self._quadrature_grid(dx=dx, num_points=num_points)
        return _overlap_matrix(self.basis_functions, xs, weights=weights)