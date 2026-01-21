
"""
coordinate_mapping.py

Build a monotone mapping u(x) from a target scale profile s(x) > 0,
so that u'(x) ~ rho(x) = k / s(x). This lets you place *uniform* (dyadic) basis
in u and get *nonuniform* spacing in x while preserving orthogonality via the
sqrt(u'(x)) Jacobian factor.

We implement a robust *piecewise-linear* mapping to avoid SciPy deps:
- Given nodes (x_i, s_i), compute rho_i = k / s_i, k chosen so that u spans [0, 1] (or [0, U]).
- Build u(x) as cumulative trapezoid of rho on the node grid, then linearly interpolate between nodes.
- Invert x(u) using numpy.interp on the monotone arrays.
- du/dx is piecewise-constant between nodes (the segment slope).

This is sufficient for preserving orthogonality analytically; numerical orthogonality
depends on your x-quadrature resolution.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Tuple, Callable
import numpy as np

@dataclass(frozen=True)
class SimpleCoordinateMap1D:
    """Monotone piecewise-linear mapping built from (x_i, s_i)."""
    x_nodes: np.ndarray   # ascending
    u_nodes: np.ndarray   # ascending, same length as x_nodes
    rho_nodes: np.ndarray # du/dx at nodes (used to compute segment slopes)

    def u_of_x(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.interp(x, self.x_nodes, self.u_nodes)

    def x_of_u(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        return np.interp(u, self.u_nodes, self.x_nodes)

    def du_dx(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        # piecewise-constant slope per segment; compute by diff of (u_nodes, x_nodes)
        xu = np.asarray(self.x_nodes, dtype=float)
        uu = np.asarray(self.u_nodes, dtype=float)
        # slopes on segments
        slopes = np.diff(uu) / np.diff(xu)  # length N-1, all > 0
        # bin x to segments
        idx = np.searchsorted(xu, x, side='right') - 1
        idx = np.clip(idx, 0, slopes.size - 1)
        return slopes[idx]


def build_mapping_from_scales(x_nodes: Sequence[float],
                              s_nodes: Sequence[float],
                              normalize_to_unit: bool = True) -> SimpleCoordinateMap1D:
    """
    Given *ascending* x_nodes and positive s_nodes (desired local "scale"/spacing),
    build u(x) so that u'(x) ~ 1 / s(x).

    Returns a SimpleCoordinateMap1D with:
      - u(x) piecewise-linear monotone
      - x(u) its linear inverse (via interpolation)
      - du/dx piecewise-constant per segment
    """
    x = np.asarray(x_nodes, dtype=float)
    s = np.asarray(s_nodes, dtype=float)
    if x.ndim != 1 or s.ndim != 1 or x.size != s.size or x.size < 2:
        raise ValueError("x_nodes and s_nodes must be 1D arrays of the same length >= 2")
    if not np.all(np.diff(x) > 0.0):
        raise ValueError("x_nodes must be strictly increasing")
    if np.any(s <= 0.0):
        raise ValueError("s_nodes must be positive")

    # target density rho ~ 1/s
    rho = 1.0 / s
    # cumulative trapezoid for u (up to a constant factor)
    u = np.empty_like(x)
    u[0] = 0.0
    diffs = np.diff(x)
    mids = 0.5 * (rho[1:] + rho[:-1])
    u[1:] = np.cumsum(mids * diffs)

    if normalize_to_unit:
        if u[-1] <= 0.0:
            raise ValueError("degenerate mapping: u span is zero")
        u = u / u[-1]  # map to [0,1]

    return SimpleCoordinateMap1D(x_nodes=x, u_nodes=u, rho_nodes=rho)
