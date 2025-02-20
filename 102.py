import math

def swish(x: float) -> float:
    x = x * (1.0 / (1.0 + math.exp(-x)))
    return round(x, 4)

for i in range(-2, 3):
    print(swish(i))
