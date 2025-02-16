import numpy as np

def phi_transform(data: list[float], degree: int) -> list[list[float]]:
    res = []
    for n in range(len(data)):
        tmp = []
        for i in range(degree + 1):
            tmp.append(data[n] ** i)
        res.append(tmp)
    return res

print(phi_transform([1.0, 2.0], 2))
