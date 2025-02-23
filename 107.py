import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    d_k = Q.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    scores = scores + mask
    attention_weights = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
    return np.matmul(attention_weights, V)

# The example input missing on the website
np.random.seed(42)
X = np.arange(48).reshape(6, 8)
X = np.random.permutation(X.flatten()).reshape(6, 8)
mask = np.triu(np.ones((6, 6)) * (-np.inf), k=1)
W_q = np.random.randint(0, 4, size=(8, 8))
W_k = np.random.randint(0, 5, size=(8, 8))
W_v = np.random.randint(0, 6, size=(8, 8))
Q, K, V = compute_qkv(X, W_q, W_k, W_v)
print(masked_attention(Q, K, V, mask))
