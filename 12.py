import numpy as np 

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    a_t = np.transpose(A)
    a_2 = a_t @ A
    v = np.eye(2)

    if a_2[0, 0] == a_2[1, 1]:
        theta = np.pi / 4
    else:
        theta = 0.5 * np.arctan2(2 * a_2[0,1], a_2[0,0] - a_2[1,1])

    r = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]]
    )
    d = np.transpose(r) @ a_2 @ r
    v = v @ r

    sigma = np.sqrt([d[0,0], d[1,1]])
    sigma_inverse = np.array([[1 / sigma[0], 0], [0, 1 / sigma[1]]])
    u = A @ v @ sigma_inverse
    return (u, sigma, v.T)

# deepml diagonal test return different sign than np.linalg.svd
print(svd_2x2_singular_values([[2, 1], [1, 2]]))
print(svd_2x2_singular_values([[1, 2], [3, 4]]))
