import numpy as np 

# btw this excercise has invalid "Expected Output" field

def svd_2x2(A: np.ndarray) -> tuple:
    A = np.array(A, dtype=np.float64)

    # eigenvectors for A^T A to get V
    Vw, V = np.linalg.eig(A.T.dot(A))
    idx = Vw.argsort()[::-1]
    Vw = Vw[idx]
    V = V[:, idx]

    w = np.sqrt(Vw)
    S = np.diag(w)

    U = A @ V @ np.diag(1 / (w + 1e-10))

    U *= -1
    V *= -1
    return U, np.diag(S), V.T

print(svd_2x2([[-10, 8], [10, -1]]))
print(svd_2x2([[2, 1], [1, 2]]))
print(svd_2x2([[1, 2], [3, 4]]))
