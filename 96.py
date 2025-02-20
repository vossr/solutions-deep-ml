def hard_sigmoid(x: float) -> float:
    if x <= -2.5:
        return 0.0
    elif x < 2.5:
        return 0.2 * x + 0.5
    else:
        return 1.0

for i in range(-9, 9):
    print(hard_sigmoid(i / 2.))
