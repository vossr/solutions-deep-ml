import numpy as np

def calculate_dot_product(vec1, vec2) -> float:
    res = 0
    for i in range(len(vec1)):
        res += vec1[i] * vec2[i]
    return res

print(calculate_dot_product(np.array([1, 2, 3]), np.array([4, 5, 6])))
