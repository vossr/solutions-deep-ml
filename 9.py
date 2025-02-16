def matrixmul(a:list[list[int|float]], b:list[list[int|float]])-> list[list[int|float]]:
    for i in range(len(a)):
        if len(a[i]) != len(b):
            return -1

    res = [[0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(a[0])):
                res[i][j] += a[i][k] * b[k][j]
    return res

A = [[1, 2],
     [2, 4]]
B = [[2, 1],
     [3, 4]]
print(matrixmul(A, B))

#                                     constant factor      Practical?
# naive                 O(n^3)        ~1                   Yes
# strassen              O(n^2.81)     ~0.67                Yes
# winograd              O(n^2.376)    10^2 to 10^3         Sometimes
# coppersmith-winograd  O(n^2.376)    10^3 to 10^6         No
# williams              O(n^2.371552) 10^6 to 10^12        No (galactic)
