def prelu(x: float, alpha: float = 0.25) -> float:
    if x > 0:
        return round(float(x), 4)
    else:
        return round(alpha * float(x), 4)

for i in range(-2, 3):
    print(prelu(i))
