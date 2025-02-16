import numpy as np

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return (np.array(matrix) * scalar).tolist()

print(scalar_multiply([[1, 2], [3, 4]], scalar = 2))
