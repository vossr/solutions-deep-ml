import numpy as np
import math

def tanh(values: np.ndarray) -> np.ndarray:
    res = np.zeros_like(values)
    for i, v in enumerate(values.flatten()):
        res.flat[i] = (math.exp(float(v)) - math.exp(float(-v))) / (math.exp(float(v)) + math.exp(float(-v)))
    return res

def tanh_derivative(values: np.ndarray) -> np.ndarray:
    res = np.zeros_like(values)
    for i, v in enumerate(values.flatten()):
        res.flat[i] = 1 - v ** 2
    return res

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

    def forward(self, x):
        hidden = np.zeros((self.hidden_size, 1))
        self.last_hiddens = [hidden]
        self.last_inputs = []
        outputs = []

        for t in range(len(x)):
            self.last_inputs.append(x[t].reshape(-1, 1))
            hidden = tanh(self.W_xh @ self.last_inputs[t] + self.W_hh @ hidden + self.b_h)
            output = self.W_hy @ hidden + self.b_y
            outputs.append(output)
            self.last_hiddens.append(hidden)
       
        self.last_outputs = outputs
        return np.array(outputs)

    def backward(self, x, y, learning_rate):
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        dW_hy = np.zeros_like(self.W_hy)
        db_h = np.zeros_like(self.b_h)
        db_y = np.zeros_like(self.b_y)

        dh_next = np.zeros((self.hidden_size, 1))

        for t in reversed(range(len(x))):
            predicted = self.last_outputs[t]
            actual = y[t].reshape(-1, 1)
            dy = predicted - actual
            dW_hy += dy @ self.last_hiddens[t+1].T
            db_y += dy

            dh = self.W_hy.T @ dy + dh_next
            dh_raw = tanh_derivative(self.last_hiddens[t + 1]) * dh

            dW_xh += dh_raw @ self.last_inputs[t].T
            dW_hh += dh_raw @ self.last_hiddens[t].T
            db_h += dh_raw

            dh_next = self.W_hh.T @ dh_raw

        self.W_xh -= learning_rate * dW_xh
        self.W_hh -= learning_rate * dW_hh
        self.W_hy -= learning_rate * dW_hy
        self.b_h -= learning_rate * db_h
        self.b_y -= learning_rate * db_y

np.random.seed(42)
input_sequence = np.array([[1.0], [2.0], [3.0], [4.0]])
expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
rnn = SimpleRNN(input_size=1, hidden_size=5, output_size=1)
output = rnn.forward(input_sequence)
rnn.backward(input_sequence, expected_output, learning_rate=0.01)
print(output)

np.random.seed(42)
input_sequence = np.array([[1.0,2.0], [7.0,2.0], [1.0,3.0], [12.0,4.0]])
expected_output = np.array([[2.0], [3.0], [4.0], [5.0]])
rnn = SimpleRNN(input_size=2, hidden_size=3, output_size=1)
for epoch in range(100):
    output = rnn.forward(input_sequence)
    rnn.backward(input_sequence, expected_output, learning_rate=0.01)
print(output)
