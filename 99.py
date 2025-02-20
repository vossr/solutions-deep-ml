import math

def softplus(x: float) -> float:
    return round(math.log(1.0 + math.exp(x)), 4)

for i in range(-2, 3):
    print(softplus(i))
