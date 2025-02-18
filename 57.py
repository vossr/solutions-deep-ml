import numpy as np

def gauss_seidel(A, b, n, x_ini=None):
    x = x_ini.copy() if x_ini is not None else np.zeros_like(b)
    for _ in range(n):
        for i in range(len(b)):
            sigma = 0.0
            for j in range(len(b)):
                if j != i:
                    sigma += A[i,j] * x[j]
            x[i] = (b[i] - sigma) / A[i,i]
    return x

A = np.array([[4, 1, 2], [3, 5, 1], [1, 1, 3]], dtype=float)
b = np.array([4, 7, 3], dtype=float)
print(gauss_seidel(A, b, 100))

A = np.array([[4, -1, 0, 1], [-1, 4, -1, 0], [0, -1, 4, -1], [1, 0, -1, 4]], dtype=float)
b = np.array([15, 10, 10, 15], dtype=float)
print(gauss_seidel(A, b, 1))
