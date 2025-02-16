def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    res = []
    for row in a:
        if (len(row) != len(b)):
            return -1
        tmp = []
        for va, vb in zip(row, b):
            tmp.append(va * vb)
        res.append(sum(tmp))
    return res

a = [[1,2],[2,4]]
b = [1,2]
print(matrix_dot_vector(a, b))
print(matrix_dot_vector([[1,2],[2,4],[6,8],[12,4]],[1,2,3]))
