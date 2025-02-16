import numpy as np

def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    B_matrix = np.array(B, dtype=float).T
    C_matrix = np.array(C, dtype=float).T

    # P = C⁻¹B
    C_inv = np.linalg.inv(C_matrix)
    P = C_inv @ B_matrix
    # get row-based transformation matrix
    P = np.transpose(P)
    return P.tolist()

B = [[1, 0, 0], 
     [0, 1, 0], 
     [0, 0, 1]]
C = [[1, 2.3, 3], 
     [4.4, 25, 6], 
     [7.4, 8, 9]]

print(transform_basis(B, C))
