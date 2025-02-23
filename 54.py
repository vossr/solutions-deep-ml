import numpy as np
import math

def tanh(values: list[float]) -> list[float]:
    res = []
    for v in values:
        activated = (math.exp(v) - math.exp(-v)) / (math.exp(v) + math.exp(-v))
        res.append(round(activated, 4))
    return res

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    current_hidden = np.array(initial_hidden_state)
    Wx = np.array(Wx)
    Wh = np.array(Wh)
    b = np.array(b)

    for x in input_sequence:
        x_array = np.array(x)
        pre_activation = np.dot(Wx, x_array) + np.dot(Wh, current_hidden) + b
        current_hidden = tanh(pre_activation)

    return [round(val, 4) for val in current_hidden]

input_sequence = [[1.0], [2.0], [3.0]]
initial_hidden_state = [0.0]
Wx = [[0.5]]
Wh = [[0.8]]
b = [0.0]

# The example outpus is wrong on the website
# The correct output is [0.9759]
print(rnn_forward(input_sequence, initial_hidden_state, Wx, Wh, b))
