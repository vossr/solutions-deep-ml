import math

def selu(x: float) -> float:
    scale = 1.0507009873554804
    x = float(x)
    if x > 0:
        return round(scale * x, 4)

    alpha = 1.6732632423543772
    return round(scale * (alpha * (math.exp(x) - 1.0)), 4)

for i in range(-10, 10):
    print(selu(i))
