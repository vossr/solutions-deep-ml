import numpy as np

# excercise 23
def softmax(scores: np.ndarray) -> np.ndarray:
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)

# excercise 53
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

def multi_head_attention(Q, K, V, n_heads):
    d_model = Q.shape[1]
    d_k = d_model // n_heads

    # reshape to n_heads, seq_len, d_k
    Q = Q.reshape(Q.shape[0], n_heads, d_k).transpose(1, 0, 2)
    K = K.reshape(K.shape[0], n_heads, d_k).transpose(1, 0, 2)
    V = V.reshape(V.shape[0], n_heads, d_k).transpose(1, 0, 2)

    attentions = []
    for i in range(n_heads):
        attn = self_attention(Q[i], K[i], V[i])
        attentions.append(attn)

    attention_output = np.concatenate(attentions, axis=-1)
    return attention_output

# The example output is wrong on the website
# The correct output is:
# [[0.73105858, 0.5],
#  [0.5, 0.73105858]]
Q = np.array([[1, 0], [0, 1]])
K = np.array([[1, 0], [0, 1]])
V = np.array([[1, 0], [0, 1]])
n_heads = 2
print(multi_head_attention(Q, K, V, n_heads))
