import math

def softmax(scores: list[float]) -> list[float]:
    res = []
    for score in scores:
        v = math.exp(score) / sum(math.exp(score) for score in scores)
        res.append(round(v, 4))
    return res

print(softmax([1, 2, 3]))
