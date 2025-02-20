import math

def sigmoid(z: float) -> float:
    return round(1 / (1 + math.exp(-z)), 4)

for z in range(-9, 10):
    print(sigmoid(z))
