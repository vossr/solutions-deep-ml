import numpy as np

def orthogonal_projection(v, L):
    v_dot_L = np.dot(v, L)
    L_dot_L = np.dot(L, L)

    if L_dot_L == 0:
        return [0] * len(v)
    scalar = v_dot_L / L_dot_L
    return [scalar * L_i for L_i in L]

v = [3, 4]
L = [1, 0]
print(orthogonal_projection(v, L))
