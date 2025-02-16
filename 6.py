def trace(matrix: list[list[float|int]]) -> float:
    res = 0
    i = 0
    while i < len(matrix) and i < len(matrix[i]):
        res += matrix[i][i]
        i += 1
    return res

def determinant2x2(matrix: list[list[float|int]]) -> list[float]:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
    tr = trace(matrix)
    dt = determinant2x2(matrix)
    discriminant = tr * tr - 4 * dt
    lambda1 = (tr + (discriminant ** 0.5)) / 2
    lambda2 = (tr - (discriminant ** 0.5)) / 2
    eigenvalues = [lambda1, lambda2]
    return eigenvalues

print(calculate_eigenvalues([[2, 1], [1, 2]]))
