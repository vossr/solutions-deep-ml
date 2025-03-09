import numpy as np
import math

# excercise 22
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

# excercise 62, 222
def tanh(values: np.ndarray) -> np.ndarray:
    res = np.zeros_like(values)
    for i, v in enumerate(values.flatten()):
        res.flat[i] = (math.exp(float(v)) - math.exp(float(-v))) / (math.exp(float(v)) + math.exp(float(-v)))
    return res

class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        hidden_state_steps = []
        hidden_state = initial_hidden_state.copy()
        cell_state = initial_cell_state.copy()

        for xt in x:
            xt = xt.reshape(self.input_size, 1)
            concat_ht_xt = np.concatenate((hidden_state, xt))
            input_gate = sigmoid(self.Wi @ concat_ht_xt + self.bi)
            forget_gate = sigmoid(self.Wf @ concat_ht_xt + self.bf)
            ct_tilde = tanh(self.Wc @ concat_ht_xt + self.bc)
            cell_state = forget_gate * cell_state + input_gate * ct_tilde
            output_gate = sigmoid(self.Wo @ concat_ht_xt + self.bo)
            hidden_state = output_gate * tanh(cell_state)
            hidden_state_steps.append(hidden_state.copy())
        return hidden_state_steps, hidden_state_steps[-1], cell_state

np.random.seed(42)
input_sequence = np.array([[1.0], [2.0], [3.0]])
initial_hidden_state = np.zeros((1, 1))
initial_cell_state = np.zeros((1, 1))
lstm = LSTM(input_size=1, hidden_size=1)
outputs, final_h, final_c = lstm.forward(input_sequence, initial_hidden_state, initial_cell_state)
print(final_h)
