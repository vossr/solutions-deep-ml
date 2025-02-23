import numpy as np
import matplotlib.pyplot as plt
import math
import time

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













def generate_linearly_separable_data(n_samples=50):
    np.random.seed(int(time.time()))
    angle1 = np.random.uniform(0, 2*np.pi)
    angle2 = (angle1 + np.pi) % (2*np.pi)

    center1 = np.array([2.5 * np.cos(angle1), 2.5 * np.sin(angle1)])
    center2 = np.array([2.5 * np.cos(angle2), 2.5 * np.sin(angle2)])

    if np.random.random() < 0.5:
        pos_center, neg_center = center1, center2
    else:
        pos_center, neg_center = center2, center1

    class1 = np.random.randn(n_samples, 2) + pos_center
    class0 = np.random.randn(n_samples, 2) + neg_center

    features = np.vstack((class1, class0))
    labels = np.array([1]*n_samples + [0]*n_samples)
    return features, labels

def plot_data_and_decision_boundary(features, labels, weights, bias):
    plt.figure(figsize=(8, 6))
    features = np.array(features)
    labels = np.array(labels)

    plt.scatter(features[labels==1, 0], features[labels==1, 1],
                color='blue', label='Class 1')
    plt.scatter(features[labels==0, 0], features[labels==0, 1],
                color='green', label='Class 0')

    x_min, x_max = plt.xlim()
    x_vals = np.linspace(x_min, x_max, 200)

    w1, w2 = weights
    if abs(w2) < 1e-7:
        vertical_x = -bias / w1
        plt.axvline(vertical_x, color='red', linestyle='--', label='Decision Boundary')
    else:
        y_vals = -(bias + w1 * x_vals) / w2
        plt.plot(x_vals, y_vals, 'r--', label='Decision Boundary')

    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Linearly Separable Data and Single Neuron Decision Boundary")
    plt.show()

def main():
    features, labels = generate_linearly_separable_data(n_samples=50)
    features_list = features.tolist()
    labels_list = labels.tolist()

    initial_weights = [0.1, -0.2]
    initial_bias = 0.0
    learning_rate = 0.1
    epochs = 100

    final_weights, final_bias, mse_values = train_neuron(features_list, labels_list,
                                             initial_weights, initial_bias,
                                             learning_rate, epochs)

    print("Final Weights:", final_weights)
    print("Final Bias:", final_bias)
    print("MSE Values:", mse_values)
    plot_data_and_decision_boundary(features, labels, final_weights, final_bias)

if __name__ == "__main__":
    main()
