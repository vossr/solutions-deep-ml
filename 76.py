import numpy as np

def cosine_similarity(v1, v2):
    return np.round(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), 3)

def cosine_similarity_slow(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude_v1 = np.sqrt(np.sum(v1 ** 2))
    magnitude_v2 = np.sqrt(np.sum(v2 ** 2))
    return np.round(dot_product / (magnitude_v1 * magnitude_v2), 3)

print(cosine_similarity([1, 2, 3], [2, 4, 6]))
