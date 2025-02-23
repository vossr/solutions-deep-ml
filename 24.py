import math

def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
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

features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
labels = [0, 1, 0]
weights = [0.7, -0.4]
bias = -0.1
print(single_neuron_model(features, labels, weights, bias))
