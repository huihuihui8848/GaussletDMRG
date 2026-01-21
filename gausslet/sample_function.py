from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt


class Function(ABC):
    """
    极简抽象基类：
    - 子类只需实现 evaluate_point(x: float) -> float | complex
    - 向量化 evaluate(x_array)、__call__、get_callable、sample、plot 全部基类提供
    """
    def __init__(self, domain: Tuple[float, float], name: Optional[str] = None) -> None:
        if not (isinstance(domain, tuple) and len(domain) == 2):
            raise ValueError("domain 必须是 (x_min, x_max) 二元组。")
        x_min, x_max = float(domain[0]), float(domain[1])
        if not (np.isfinite(x_min) and np.isfinite(x_max)) or x_max <= x_min:
            raise ValueError("需要有限端点且满足 x_max > x_min。")
        self._domain = (x_min, x_max)
        self._name = name or self.__class__.__name__

    # ---- 子类只需实现这一个方法（标量版）----
    @abstractmethod
    def evaluate_point(self, x: float):
        """返回 f(x) 的标量值（可实可复）。"""
        raise NotImplementedError

    # ---- 基类自动提供的向量化接口 ----
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """向量化包装，子类无需关心。"""
        x = np.asarray(x)
        f_vec = np.vectorize(self.evaluate_point, otypes=[np.complex128])
        y = f_vec(x)
        # 如果都是实数，挤掉虚部
        if np.allclose(y.imag, 0.0):
            y = y.real
        return y

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)

    def get_callable(self) -> Callable[[np.ndarray], np.ndarray]:
        return lambda x: self.evaluate(x)

    # ---- 实用方法 ----
    @property
    def domain(self) -> Tuple[float, float]:
        return self._domain

    @property
    def name(self) -> str:
        return self._name

    def sample(self, n: int = 1024, endpoint: bool = True):
        x_min, x_max = self._domain
        x = np.linspace(x_min, x_max, int(n), endpoint=endpoint)
        return x, self.evaluate(x)

    def plot(
        self,
        n: int = 1024,
        endpoint: bool = True,
        *,
        mode: str = "auto",  # 'auto'|'real'|'imag'|'abs'|'phase'
        title: Optional[str] = None,
        show: bool = True,
    ):
        x, y = self.sample(n=n, endpoint=endpoint)
        # 复数可视化选择
        if np.iscomplexobj(y):
            if mode == "auto":
                # 若实部主导则画实部，否则画 |y|
                y_plot = y.real if np.allclose(y.imag, 0.0) else np.abs(y)
                y_label = "Re f(x)" if y_plot is y.real else "|f(x)|"
            elif mode == "real":
                y_plot, y_label = y.real, "Re f(x)"
            elif mode == "imag":
                y_plot, y_label = y.imag, "Im f(x)"
            elif mode == "abs":
                y_plot, y_label = np.abs(y), "|f(x)|"
            elif mode == "phase":
                y_plot, y_label = np.angle(y), "arg f(x)"
            else:
                raise ValueError("mode 必须是 'auto'|'real'|'imag'|'abs'|'phase'")
        else:
            y_plot, y_label = y, "f(x)"

        plt.figure(figsize=(8, 4.2))
        plt.plot(x, y_plot)  # 不指定颜色
        plt.xlabel("x")
        plt.ylabel(y_label)
        plt.title(title if title is not None else self.name)
        plt.tight_layout()
        if show:
            plt.show()

    def __repr__(self) -> str:
        x_min, x_max = self._domain
        return f"{self.name}(domain=({x_min}, {x_max}))"


class DoubleGaussianWell(Function):
    def __init__(
        self,
        domain=(0.0, 10.0),
        *,
        c1=3.0, fwhm1=0.6, depth1=8.0,
        c2=7.0, fwhm2=1.2, depth2=4.0,
        background=0.0
    ):
        super().__init__(domain, name="DoubleGaussianWell")
        self.c1, self.c2 = float(c1), float(c2)
        self.s1 = fwhm1 / (2.0*np.sqrt(2.0*np.log(2.0)))
        self.s2 = fwhm2 / (2.0*np.sqrt(2.0*np.log(2.0)))
        self.d1, self.d2 = float(depth1), float(depth2)
        self.bg = float(background)

    def evaluate_point(self, x: float):
        w1 = np.exp(-0.5*((x - self.c1)/self.s1)**2)
        w2 = np.exp(-0.5*((x - self.c2)/self.s2)**2)
        return self.bg - self.d1*w1 - self.d2*w2


class SineBackgroundWithGaussians(Function):
    """
    f(x) = offset + A_sin * sin(2π f x + phi)
           + A1 * exp(-0.5 * ((x - c1)/sigma1)^2)
           + A2 * exp(-0.5 * ((x - c2)/sigma2)^2)
    """
    def __init__(
        self,
        domain=(0.0, 10.0),
        *,
        # sin background
        A_sin=1.0,
        freq=0.5,
        phi=0.0,
        offset=0.0,
        # Gaussian 1
        c1=3.0,
        fwhm1=0.6,
        A1=-0.8,
        # Gaussian 2
        c2=7.0,
        fwhm2=1.2,
        A2=-0.4,
        name="SineBackgroundWithGaussians",
    ):
        super().__init__(domain, name=name)
        self.A_sin  = float(A_sin)
        self.freq   = float(freq)
        self.phi    = float(phi)
        self.offset = float(offset)

        self.c1, self.c2 = float(c1), float(c2)
        # FWHM -> sigma
        self.s1 = float(fwhm1) / (2.0*np.sqrt(2.0*np.log(2.0)))
        self.s2 = float(fwhm2) / (2.0*np.sqrt(2.0*np.log(2.0)))
        if self.s1 <= 0 or self.s2 <= 0:
            raise ValueError("fwhm1/fwhm2 必须为正。")

        self.A1 = float(A1)
        self.A2 = float(A2)

    def evaluate_point(self, x: float):
        bg = self.offset + self.A_sin * np.sin(2*np.pi*self.freq*x + self.phi)
        g1 = self.A1 * np.exp(-0.5 * ((x - self.c1)/self.s1)**2)
        g2 = self.A2 * np.exp(-0.5 * ((x - self.c2)/self.s2)**2)
        return bg + g1 + g2


from gausslet import gausslet_scaled
class Gausslet(Function):

    def __init__(self, coes, scale=1, center=0, decay=3, name = "Gausslet"):
        domain = (center-scale*decay, center+scale*decay)
        super().__init__(domain, name)
        self.coes = np.asarray(coes, dtype=float).copy()
        self.scale = scale
        self.center = center
    
    def evaluate_point(self, x):
        return gausslet_scaled(x, self.coes, self.center, self.scale)
    
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return gausslet_scaled(x, self.coes, self.center, self.scale)





if __name__ == '__main__':
    V = DoubleGaussianWell((0.0, 10.0), c1=2.8, fwhm1=0.5, depth1=9.0,
                                       c2=7.2, fwhm2=1.2, depth2=4.5)
    V.plot(n=1500, title="Double Gaussian Well")

    _, coe = np.loadtxt("coefficient/G4.csv", delimiter=",", unpack=True)
    gausslet = Gausslet(coe)
    gausslet.plot()