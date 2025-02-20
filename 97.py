import math

def elu(x: float, alpha: float = 1.0) -> float:
    if x > 0:
        val = float(x)
    else:
        val = alpha * (math.exp(x) - 1.0)
    return round(val, 4)

for i in range(-2, 5):
    print(elu(i))
