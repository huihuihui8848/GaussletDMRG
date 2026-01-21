import numpy as np

def read_coef(path):
    _, b = np.loadtxt(path, unpack=True, delimiter=',')
    return b

# Calculate centers and scale for given L and N
def make_centers_and_scale(L, N):
    """
    Generate:
        centers: [s_min, ..., -1s, 0, 1s, ..., s_max] centers for each gausslet
        scales : s = L / N for every basis function

    Requirement:
        N = 1 + 2*M must be odd.
    """
    if N % 2 == 0:
        raise ValueError(
            f"N = {N} is even. For symmetric arrangement 0, +/-2s, ..., "
            "the number of basis functions N must be odd (N = 1 + 2*M)."
        )

    s = L / N
    centers = np.zeros(N, dtype=float)

    M = (N - 1) // 2             # There are M pairs of +/-
    for i in range(1, M + 1):
        centers[2 * i - 1] = -2 * i * s   # Odd index: negative
        centers[2 * i]     =  2 * i * s   # Even index: positive
    centers.sort()

    # scales = np.full(N, s, dtype=float)   # Same scale for all
    return centers, s
