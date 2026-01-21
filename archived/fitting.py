from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

from ..gausslet.gausslet import gausslet_scaled

__all__ = [
    "project_onto_grid_gausslets",
    "reconstruct_from_coeffs",
]


def _centers_from_interval(
    interval: tuple[float, float],
    bcoef: Sequence[float],
    s: float,
    pad_sigma: float = 6.0,
) -> np.ndarray:
    """Return a grid of gausslet centers that comfortably covers the interval."""
    left, right = float(interval[0]), float(interval[1])
    bcoef = np.asarray(bcoef, dtype=float)
    J = len(bcoef) - 1
    sigma_x = s / 3.0
    far_internal = (J / 3.0) * s
    margin = far_internal + pad_sigma * sigma_x

    k0 = int(np.floor((left - margin) / s))
    k1 = int(np.ceil((right + margin) / s))
    return s * np.arange(k0, k1 + 1, dtype=float)


def _trapezoidal_weights(xs: np.ndarray) -> np.ndarray:
    if xs.ndim != 1:
        raise ValueError("quadrature grid must be one-dimensional")
    if xs.size < 2:
        raise ValueError("quadrature grid must contain at least two points")

    diffs = np.diff(xs)
    if np.any(diffs <= 0.0):
        raise ValueError("quadrature grid must be strictly increasing")

    weights = np.empty_like(xs, dtype=float)
    weights[0] = 0.5 * diffs[0]
    weights[-1] = 0.5 * diffs[-1]
    if xs.size > 2:
        weights[1:-1] = 0.5 * (diffs[:-1] + diffs[1:])
    return weights


def _quadrature_grid_for_interval(
    interval: tuple[float, float],
    bcoef: Sequence[float],
    s: float,
    dx: float | None,
    pad_sigma: float = 6.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate quadrature abscissae and trapezoidal weights for the fit."""
    left, right = float(interval[0]), float(interval[1])
    bcoef = np.asarray(bcoef, dtype=float)
    J = len(bcoef) - 1
    sigma_x = s / 3.0
    far_internal = (J / 3.0) * s
    margin = far_internal + pad_sigma * sigma_x

    if dx is None:
        dx = s / 200.0

    start = left - margin
    stop = right + margin
    xs = np.arange(start, stop + dx, dx, dtype=float)
    weights = _trapezoidal_weights(xs)
    return xs, weights


def project_onto_grid_gausslets(
    f: Callable[[np.ndarray], np.ndarray],
    bcoef: Sequence[float],
    s: float = 1.0,
    interval: tuple[float, float] = (-4.0, 4.0),
    centers: Iterable[float] | None = None,
    dx: float | None = None,
    reg: float = 0.0,
    pad_sigma: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Least-squares projection of ``f`` onto translated/scaled gausslets."""
    bcoef = np.asarray(bcoef, dtype=float)

    if centers is None:
        centers = _centers_from_interval(interval, bcoef, s, pad_sigma=pad_sigma)
    centers = np.asarray(list(centers), dtype=float)

    xs, weights = _quadrature_grid_for_interval(interval, bcoef, s, dx, pad_sigma=pad_sigma)
    basis_columns = [gausslet_scaled(xs, bcoef, center=c, s=s) for c in centers]
    B = np.stack(basis_columns, axis=1)

    f_vals = np.asarray(f(xs), dtype=float)
    if f_vals.shape != xs.shape:
        raise ValueError("f(xs) must return an array matching the quadrature grid shape")

    W = weights[:, None]
    S = B.T @ (B * W)
    rhs = B.T @ (f_vals * weights)

    if reg > 0.0:
        S = S + reg * np.eye(S.shape[0], dtype=S.dtype)

    try:
        coeffs = np.linalg.solve(S, rhs)
    except np.linalg.LinAlgError:
        coeffs, *_ = np.linalg.lstsq(S, rhs, rcond=None)

    f_rec = B @ coeffs
    l2_error = float(np.sqrt(np.sum((f_rec - f_vals) ** 2 * weights)))

    try:
        cond_S = float(np.linalg.cond(S))
    except np.linalg.LinAlgError:
        cond_S = float(np.inf)

    return coeffs, centers, cond_S, l2_error


def reconstruct_from_coeffs(
    x: Sequence[float] | np.ndarray,
    coeffs: Sequence[float],
    centers: Sequence[float],
    bcoef: Sequence[float],
    s: float = 1.0,
) -> np.ndarray:
    """Reconstruct function values on ``x`` from fitted gausslet coefficients."""
    x = np.asarray(x, dtype=float)
    coeffs = np.asarray(coeffs, dtype=float)
    centers = np.asarray(centers, dtype=float)
    bcoef = np.asarray(bcoef, dtype=float)

    columns = [gausslet_scaled(x, bcoef, center=c, s=s) for c in centers]
    Phi = np.stack(columns, axis=1)
    return Phi @ coeffs
