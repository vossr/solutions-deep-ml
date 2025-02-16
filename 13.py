import numpy as np

def determinant_4x4(matrix: list[list[float]]) -> float:
    size = len(matrix)
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for col in range(size):
        # remove first row
        submatrix = np.delete(matrix, 0, axis=0)
        # remove current column
        minor = np.delete(submatrix, col, axis=1).tolist()
        cofactor = ((-1) ** col) * matrix[0][col] * determinant_4x4(minor)
        det += cofactor
    return det

print(determinant_4x4([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))
print(determinant_4x4([[4, 3, 2, 1], [3, 2, 1, 4], [2, 1, 4, 3], [1, 4, 3, 2]]))
