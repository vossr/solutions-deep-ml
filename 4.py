def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "row":
        return [sum(row) / len(row) for row in matrix]
    elif mode == "column":
        matrix = [list(col) for col in zip(*matrix)]# transpose
        return [sum(row) / len(row) for row in matrix]
    else:
        return []

print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'))
print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'row'))
