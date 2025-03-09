import numpy as np

def softmax(scores: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

def self_attention(Q, K, V):
    d_k = K.shape[1]
    attention_scores = Q @ K.T
    attention_scores = np.float64(attention_scores)
    attention_scores /= np.sqrt(d_k)
    attention_weights = softmax(attention_scores)
    attention_output = attention_weights @ V
    return attention_output

def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)
print(output)
