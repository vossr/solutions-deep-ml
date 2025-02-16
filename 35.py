import numpy as np#not needed

def make_diagonal(x):
    res = [[0.0 for _ in range(len(x))] for _ in range(len(x))]
    for i, val in enumerate(x):
        res[i][i] = float(val)
    return res

print(make_diagonal([1, 2, 3]))
