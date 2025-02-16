import numpy as np

def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    return np.round(np.linalg.inv(matrix), decimals=10).tolist()

print(inverse_2x2([[4, 7], [2, 6]]))
