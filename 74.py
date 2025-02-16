import numpy as np

def create_row_hv(row:dict[str,str], dim:int, random_seeds:dict[str,int]):
    row_hvs = []
    for col in row.keys():
        np.random.seed(random_seeds[col])
        a = np.random.choice([-1, 1], dim)
        b = np.random.choice([-1, 1], dim)
        row_hvs.append(a * b)
    res = np.sum(row_hvs, axis=0)
    res = np.array([1 if v >= 0 else -1 for v in res])
    return res

row = {"FeatureA": "value1", "FeatureB": "value2"}
random_seeds = {"FeatureA": 42, "FeatureB": 7}
dim = 5
print(create_row_hv(row, dim, random_seeds))
