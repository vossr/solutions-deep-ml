import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    augmented = np.hstack((A, b.reshape(-1, 1)))

    for i in range(n):
        max_row = np.argmax(np.abs(augmented[i:, i])) + i
        augmented[[i, max_row]] = augmented[[max_row, i]]
        for j in range(i+1, n):
            factor = augmented[j, i] / augmented[i, i]
            augmented[j, i:] -= factor * augmented[i, i:]

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (augmented[i, -1] - augmented[i, i+1:n].dot(x[i+1:])) / augmented[i, i]
    return x

A = np.array([[2, 8, 4], [2, 5, 1], [4, 10, -1]], dtype=float)
b = np.array([2, 5, 1], dtype=float)
print(gaussian_elimination(A, b))
