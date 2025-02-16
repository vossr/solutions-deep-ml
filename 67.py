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

def compressed_col_sparse_matrix(dense_matrix):
    return compressed_row_sparse_matrix(np.transpose(dense_matrix))

print(compressed_col_sparse_matrix([
    [0, 0, 3, 0],
    [1, 0, 0, 4],
    [0, 2, 0, 0]
]))
