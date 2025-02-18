import numpy as np

def conjugate_gradient(A, b, n, x0=None, tol=1e-8):
    if x0 is None:
        x = np.zeros_like(b)
    else:
        x = x0.copy()
    r = b - A @ x
    p = r.copy()

    for _ in range(n):
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        if np.linalg.norm(r_new) < tol:
            break
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
    return x

A = np.array([[4, 1], [1, 3]])
b = np.array([1, 2])
print(conjugate_gradient(A, b, 5))
