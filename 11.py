import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    A = np.array(A)
    x = np.zeros_like(b, dtype=np.float64)

    for _ in range(n):
        x_new = np.zeros_like(x)
        for i in range(A.shape[0]):
            s = 0
            for j in range(A.shape[1]):
                if i != j:
                    s += A[i, j] * x[j]
            x_new[i] = (b[i] - s) / A[i, i]
        x = x_new
    return x.tolist()

# iteratively solve A * what = b
A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
b = [-1, 2, 3]
n = 2
print([f"{xi:.4f}" for xi in solve_jacobi(A, b, n)])
