import numpy as np

# from excercise 48
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

def matrix_image(A):
    mrref = rref(A)

    rows, cols = mrref.shape
    pivot_cols = []
    row_idx = 0
    for col in range(cols):
        if row_idx < rows:
            if abs(mrref[row_idx][col]) > 1e-10:
                # has pivot
                pivot_cols.append(col)
                row_idx += 1

    return A[:, pivot_cols]

print(matrix_image(np.array([
[1, 2, 3],
[4, 5, 6],
[7, 8, 9]
])))
