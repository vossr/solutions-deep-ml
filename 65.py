import numpy as np

def compressed_row_sparse_matrix(dense_matrix):
    vals = []
    col_idx = []
    row_ptr = [0]

    for y in range(len(dense_matrix)):
        nonzeros_in_row = 0
        for x in range(len(dense_matrix[y])):
            if dense_matrix[y][x] != 0:
                vals.append(dense_matrix[y][x])
                col_idx.append(x)
                nonzeros_in_row += 1
        row_ptr.append(row_ptr[-1] + nonzeros_in_row)
    return vals, col_idx, row_ptr

dense_matrix = [
    [1, 0, 0, 0],
    [0, 2, 0, 0],
    [3, 0, 4, 0],
    [1, 0, 0, 5]
]

vals, col_idx, row_ptr = compressed_row_sparse_matrix(dense_matrix)
print("Values array:", vals)
print("Column indices array:", col_idx)
print("Row pointer array:", row_ptr)
