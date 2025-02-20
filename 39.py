import numpy as np
import math

# Directly applying the logarithm to the softmax function can lead to numerical instability
# So use the log-sum-exp trick

def log_softmax(scores: list) -> np.ndarray:
    max_score = max(scores)
    sum_exp = sum(math.exp(score - max_score) for score in scores)
    res = []
    for score in scores:
        v = score - max_score - math.log(sum_exp)
        res.append(round(v, 4))
    return np.array(res)

A = np.array([1, 2, 3])
print(log_softmax(A))
