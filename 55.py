import numpy as np

def translate_object(points, tx, ty):
    homogeneous_points = np.array([[p[0], p[1], 1] for p in points])

    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])
    translated_points = np.matmul(homogeneous_points, translation_matrix.T)
    return translated_points[:, :2].tolist()

print(translate_object([[0, 0], [1, 0], [0.5, 1]], 2, 3))
