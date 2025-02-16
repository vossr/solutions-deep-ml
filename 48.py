import numpy as np

def rref(matrix):
    matrix = matrix.astype(float)
    rows, cols = matrix.shape
    x = 0

    for y in range(rows):
        if x >= cols:
            break

        # find pivot column
        pivot_row = y
        while pivot_row < rows and matrix[pivot_row][x] == 0:
            pivot_row += 1

        if pivot_row == rows:
            x += 1
            y -= 1
            continue

        if pivot_row != y:
            # swap rows
            matrix[[pivot_row, y]] = matrix[[y, pivot_row]]

        # normalize pivot row
        pivot_val = matrix[y][x]
        matrix[y] = matrix[y] / pivot_val

        # eliminate other rows
        for i in range(rows):
            if i != y:
                factor = matrix[i][x]
                matrix[i] -= factor * matrix[y]
        x += 1
    return matrix

matrix = np.array([
    [1, 2, -1, -4],
    [2, 3, -1, -11],
    [-2, 0, -3, 22]
])
print(rref(matrix))
