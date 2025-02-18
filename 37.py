import numpy as np

def calculate_correlation_matrix(X, Y=None):
    if Y is None:
        Y = X

    cm = np.zeros((X.shape[1], Y.shape[1]))
    for i in range(X.shape[1]):
        for j in range(Y.shape[1]):
            x = X[:, i]
            y = Y[:, j]
            x_mean = np.mean(x)
            y_mean = np.mean(y)

            numerator = np.sum((x - x_mean) * (y - y_mean))
            x_std = np.sqrt(np.sum((x - x_mean) ** 2))
            y_std = np.sqrt(np.sum((y - y_mean) ** 2))

            # correlation coefficient
            if x_std != 0 and y_std != 0:
                cm[i, j] = numerator / (x_std * y_std)
    return cm

print(calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])))
