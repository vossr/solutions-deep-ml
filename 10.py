import numpy as np

def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    num_features = len(vectors)
    num_samples = len(vectors[0])
    means = [np.average(feature) for feature in vectors]
    covariance_matrix = [[0.0 for _ in range(num_features)] for _ in range(num_features)]

    for i in range(num_features):
        for j in range(num_features):
            covariance = 0.0
            for k in range(num_samples):
                covariance += (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
            covariance /= (num_samples - 1)
            covariance_matrix[i][j] = covariance
    return covariance_matrix

print(calculate_covariance_matrix([[1, 2, 3], [4, 5, 6]]))
print(calculate_covariance_matrix([[1, 5, 6], [2, 3, 4], [7, 8, 9]]))
