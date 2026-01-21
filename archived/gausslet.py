from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Iterable, List
import numpy as np

# ---- Coefficient tables (bj) for primary gausslets ----
# Table III (G6) coefficients (j >= 0). Symmetry: b[-j] = b[j].
_G6_BJ: List[float] = [
    0.6188489361270065,
    0.3824167454273702,
    0.1099474897465580,
   -0.1478654707279702,
   -0.1092533175894797,
    0.0008350876805188,
    0.0383468513752624,
    0.0443793867348271,
   -0.0264220705098279,
    0.0039445390490703,
   -0.0180915921775044,
    0.0130345989975900,
   -0.0046547275254753,
    0.0040100201083541,
   -0.0015184700034549,
   -0.0002473520884477,
    0.0004021212441880,
   -0.0006709855121816,
    0.0006479929834526,
   -0.0004072275423943,
    0.0002507059396952,
   -0.0001490130367618,
    0.0000852936240908,
   -0.0000498804314109,
    0.0000289020341125,
   -0.0000168829682435,
    0.0000097119084642,
   -0.0000054294948468,
    0.0000027905622899,
   -0.0000011656692668,
    0.0000004251304827,
   -0.0000001918178974,
    0.0000001011853914,
   -0.0000000521084195,
    0.0000000266388167,
   -0.0000000129272754,
    0.0000000067016738,
   -0.0000000032075046,
    0.0000000009153639,
   -0.0000000001508133,
    0.0000000000575772,
   -0.0000000000147799,
   -0.0000000000041069,
    0.0000000000015572,
    0.0000000000000007,
    0.0000000000000002,
    0.0000000000000001,
    0.0000000000000002,
    0.0000000000000000,
]

def gausslet_coeffs(order: str = "G6") -> List[float]:
    order = order.upper()
    if order == "G6":
        return _G6_BJ.copy()
    raise ValueError(f"Unsupported order '{order}'. Supported: 'G6'.")

def _underlying_gaussian(arg: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * arg * arg)

def gausslet_primary(x: np.ndarray, order: str = "G6") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    bj = gausslet_coeffs(order)
    J = len(bj) - 1
    y = bj[0] * _underlying_gaussian(3.0 * x - 0.0)
    for j in range(1, J + 1):
        b = bj[j]
        if b == 0.0:
            continue
        y += b * (_underlying_gaussian(3.0 * x - j) + _underlying_gaussian(3.0 * x + j))
    return y

def gausslet_scaled(x: np.ndarray, center: float = 0.0, s: float = 1.0, order: str = "G6") -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return (1.0 / np.sqrt(s)) * gausslet_primary((x - center) / s, order=order)

def build_basis(centers: Iterable[float], s: float = 1.0, order: str = "G6"):
    centers = list(centers)
    def make_fun(c: float):
        return lambda x: gausslet_scaled(x, center=c, s=s, order=order)
    return [make_fun(c) for c in centers]

def overlap_matrix(centers: Iterable[float], s: float = 1.0, order: str = "G6",
                   L: float | None = None, dx: float = 1e-3) -> np.ndarray:
    centers = np.array(list(centers), dtype=float)
    n = len(centers)
    if L is None:
        J = len(gausslet_coeffs(order)) - 1
        pad = 2.5 * s
        L = (np.max(np.abs(centers)) + (J / 3.0) * s / 2.0) + pad
    xs = np.arange(-L, L + dx, dx)
    B = [gausslet_scaled(xs, center=c, s=s, order=order) for c in centers]
    B = np.stack(B, axis=1)
    S = dx * (B.T @ B)
    return S

if __name__ == '__main__':
    centers = np.arange(-5,5,1.2)
    print(overlap_matrix(centers))
