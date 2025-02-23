import math

def tanh(values: list[float]) -> list[float]:
    res = []
    for v in values:
        activated = (math.exp(v) - math.exp(-v)) / (math.exp(v) + math.exp(-v))
        res.append(round(activated, 4))
    return res

print(tanh([-2, 0, 2])) 
