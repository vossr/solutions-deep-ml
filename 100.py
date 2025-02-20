def softsign(x: float) -> float:
    x = float(x)
    return round(x / (1.0 + abs(x)), 4)

for i in range(-2, 3):
    print(softsign(i))
