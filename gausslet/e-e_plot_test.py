import numpy as np
import matplotlib.pyplot as plt


def load_V(filename):
    data = np.load(filename)
    V = data["V"]
    return V, data



def scatter_V_by_legs(V, legA="i", legB="j", max_points=None, ymin=1e-16):
    """
    V      : V[i,j,k,l]
    legA   : "i","j","k","l"
    legB   : "i","j","k","l"
    max_points : set a cap for maximum point, exceed the cap will random choose the scatter
    """

    # map "i","j","k","l" to 0,1,2,3
    LEG_MAP = {"i": 0, "j": 1, "k": 2, "l": 3}
    a = LEG_MAP[legA]
    b = LEG_MAP[legB]

    # 生成索引网格: idx 的 shape = (4, N, N, N, N)
    # idx[0] = i 网格, idx[1] = j 网格, ...
    idx = np.indices(V.shape)

    # 对应两个 leg 的索引之差的绝对值
    dist = np.abs(idx[a] - idx[b])     # shape (N,N,N,N)

    # 展平成一维
    x = dist.ravel()
    # 万一 V 是 complex，就取实部；如果肯定是实数也没问题
    y = np.abs(np.real(V)).ravel()
    y[y < ymin] = ymin

    # 如果点数太多，可以随机抽样
    if max_points is not None and max_points < x.size:
        sel = np.random.choice(x.size, size=max_points, replace=False)
        x = x[sel]
        y = y[sel]

    # 画图
    plt.figure()
    plt.scatter(x, y, s=5, alpha=0.4)
    plt.xlabel(f"|{legA} - {legB}|")
    plt.ylabel("V elements")
    plt.title(f"V elements vs |{legA}-{legB}|")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    #plt.savefig(f'img/Coulomb/{legA}-{legB}.png')
    plt.show()


if __name__ == "__main__":
    V, meta = load_V("data/V.npz") 

    # |i-j|
    scatter_V_by_legs(V, legA="i", legB="j", max_points=200000)
    # |i-k|
    scatter_V_by_legs(V, legA="i", legB="k", max_points=200000)
    # |i-l|
    scatter_V_by_legs(V, legA="i", legB="l", max_points=200000)
