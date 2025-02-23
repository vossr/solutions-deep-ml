import numpy as np

def softmax(scores: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def pattern_weaver(n, crystal_values, dimension):
    X = np.array(crystal_values).reshape(-1, 1)
    attention_scores = (X @ X.T) / dimension
    attention_weights = softmax(attention_scores)
    enhanced_pattern = attention_weights @ X
    return np.round(enhanced_pattern.flatten(), 3)

print(pattern_weaver(5, [4, 2, 7, 1, 9], 1))
