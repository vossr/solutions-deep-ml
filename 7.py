import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        return -1
    return np.round(np.matmul(np.matmul(np.linalg.inv(T), A), S), decimals=10).tolist()

A = [[1, 2], [3, 4]]
T = [[2, 0], [0, 2]]
S = [[1, 1], [0, 1]]
print(transform_matrix(A, T, S))
print(transform_matrix([[1, 0], [0, 1]], [[1, 2], [3, 4]], [[2, 0], [0, 2]]))
