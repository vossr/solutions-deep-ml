import numpy as np
import math

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def sigmoid_derivative(z: float) -> float:
    return z * (1 - z)

def single_neuron_forward_pass(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    probabilities = []
    for i in range(len(features)):
        z = 0
        for k in range(2):
            z += features[i][k] * weights[k]
        z += bias
        probability = sigmoid(z)
        probabilities.append(probability)
    tmp = [(v - true) ** 2 for v, true in zip(probabilities, labels)]
    mse = sum(tmp) / len(tmp)
    return [round(p, 4) for p in probabilities], round(mse, 4)

def backpropagate(features: list[list[float]], labels: list[int], probabilities: list[float]) -> (list[float], float):
    dw = [0, 0]
    db = 0

    for i in range(len(features)):
        dl_dp = 2 * (probabilities[i] - labels[i])
        dp_dz = sigmoid_derivative(probabilities[i])
        dl_dz = dl_dp * dp_dz

        for k in range(2):
            dw[k] += dl_dz * features[i][k]
        db += dl_dz

    dw = [grad / len(features) for grad in dw]
    db = db / len(features)
    return dw, db

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    weights = initial_weights.copy()
    bias = initial_bias
    mse_values = []

    for epoch in range(epochs):
        probabilities, mse = single_neuron_forward_pass(features, labels, weights, bias)
        mse_values.append(mse)

        dw, db = backpropagate(features, labels, probabilities)

        weights = [w - learning_rate * dw_i for w, dw_i in zip(weights, dw)]
        bias = bias - learning_rate * db

    weights = [round(w, 4) for w in weights]
    mse_values = [round(mse, 4) for mse in mse_values]
    return weights, round(bias, 4), mse_values

features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]
labels = [1, 0, 0]
initial_weights = [0.1, -0.2]
initial_bias = 0.0
learning_rate = 0.1
epochs = 2
print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))
