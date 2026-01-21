import numpy as np
import matplotlib.pyplot as plt


def load_V(filename):
    data = np.load(filename)
    V = data["V"]
    return V, data


def scatter_V_by_legs(
    V,
    legA="i",
    legB="j",
    max_points=None,
    ymin=1e-16,
    branch=None,
    max_branch=None,
):
    """
    V      : V[i,j,k,l]
    legA   : "i","j","k","l"   -> 用来画横轴 |legA - legB|
    legB   : "i","j","k","l"
    max_points : 最大点数，超出后随机抽样
    ymin   : log y 轴的下限
    branch : None 或者 ("i","j") 这类的 tuple
             当不为 None 时，用 |branch[0] - branch[1]| 做“分支”着色
             例如 branch=("i","j")，画 |i-k| 的同时按 |i-j| 颜色区分
    max_branch : 只单独着色到这个 branch 值，
                 其余的合并到一个 ">=max_branch" 分支中
    """

    # map "i","j","k","l" to 0,1,2,3
    LEG_MAP = {"i": 0, "j": 1, "k": 2, "l": 3}
    a = LEG_MAP[legA]
    b = LEG_MAP[legB]

    # 生成索引网格: idx 的 shape = (4, N, N, N, N)
    idx = np.indices(V.shape)

    # 主轴距离 |legA - legB|
    dist_main = np.abs(idx[a] - idx[b])  # shape (N,N,N,N)

    # 展平成一维
    x = dist_main.ravel()
    y = np.abs(np.real(V)).ravel()
    y[y < ymin] = ymin

    # 如果要根据 branch 分颜色，就计算 branch 的距离
    if branch is not None:
        c = LEG_MAP[branch[0]]
        d = LEG_MAP[branch[1]]
        branch_dist = np.abs(idx[c] - idx[d]).ravel()
    else:
        branch_dist = None

    # 先统一做抽样（如果需要），保证 x/y/branch_dist 一起被抽样
    if max_points is not None and max_points < x.size:
        sel = np.random.choice(x.size, size=max_points, replace=False)
        x = x[sel]
        y = y[sel]
        if branch_dist is not None:
            branch_dist = branch_dist[sel]

    plt.figure()

    if branch_dist is None:
        # 和原来一样，不分颜色
        plt.scatter(x, y, s=5, alpha=0.4)
    else:
        # 有 branch 的情况，用不同颜色画不同的分支
        unique_branches = np.unique(branch_dist)

        # 如果设置了 max_branch，就把 >=max_branch 的合并
        if max_branch is not None:
            small = unique_branches[unique_branches < max_branch]
            has_big = np.any(unique_branches >= max_branch)
            branches_to_plot = list(small)
            if has_big:
                branches_to_plot.append(f">={max_branch}")
        else:
            branches_to_plot = list(unique_branches)

        # 构造 colormap
        import matplotlib.cm as cm

        cmap = cm.get_cmap("tab10", len(branches_to_plot))

        for idx_branch, bval in enumerate(branches_to_plot):
            if isinstance(bval, str):
                # 这个是 >=max_branch 的合并分支
                mask = branch_dist >= max_branch
                label = f"|{branch[0]}-{branch[1]}|>={max_branch}"
            else:
                mask = branch_dist == bval
                label = f"|{branch[0]}-{branch[1]}|={int(bval)}"

            if not np.any(mask):
                continue

            plt.scatter(
                x[mask],
                y[mask],
                s=5,
                alpha=0.8,
                color=cmap(idx_branch),
                label=label,
            )

        plt.legend()

    plt.xlabel(f"|{legA} - {legB}|")
    plt.ylabel("V elements")
    plt.title(f"V elements vs |{legA}-{legB}|")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    V, meta = load_V("data/V.npz")

    scatter_V_by_legs(
        V,
        legA="i",
        legB="k",
        branch=("i", "j"),
        max_points=200000,
        max_branch=5
    )

    scatter_V_by_legs(
        V,
        legA="i",
        legB="k",
        branch=("i", "l"),
        max_points=200000,
        max_branch=5
    )
