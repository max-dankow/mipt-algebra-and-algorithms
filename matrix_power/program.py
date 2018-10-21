import numpy as np
import sys
import time


def read_matrix():
    n = 0
    numbers = []
    for line in sys.stdin.readlines():
        row = [int(x) for x in line.split()]
        numbers = numbers + row
        n = len(row)

    return np.array(numbers).reshape((n, n))


def print_matrix(matrix):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if col == matrix.shape[1] - 1:
                print(matrix[row][col], end='')
            else:
                print(matrix[row][col], end=' ')
        print()


def log_power(x, power, multiply):
    if power == 0:
        return 1
    if power == 1:
        return x

    rest = log_power(x, power // 2, multiply)
    result = multiply(rest, rest)
    if power % 2 == 1:
        result = multiply(result, x)
    return result


def multiply_matrices(a, b, mod=9):
    assert(a.shape == b.shape)
    n = a.shape[0]
    new_size = __complete_to_power(n, base=2)
    a_new = __complete_matrix(a, (new_size, new_size))
    b_new = __complete_matrix(b, (new_size, new_size))
    result = mmul(a_new, b_new, mod)
    return result[0:n, 0:n]


def simple_matmul(a, b):
    ab = np.empty(shape=(a.shape[0], b.shape[1]), dtype=int)
    for i in range(a.shape[0]):
        for k in range(b.shape[1]):
            ab[i, k] = np.dot(a[i, :], b[:, k])
    return ab


def mmul(a, b, mod, strassen_max_size=64):
    # print('mmul', a.shape[0])
    # assert(a.shape == b.shape)
    if (a.shape[0] <= strassen_max_size):
        return simple_matmul(a, b) % mod

    a11, a12, a21, a22 = __disassemble_matrix(a)
    b11, b12, b21, b22 = __disassemble_matrix(b)
    m1 = mmul(a11 + a22, b11 + b22, mod)
    m2 = mmul(a21 + a22, b11, mod)
    m3 = mmul(a11, b12 - b22, mod)
    m4 = mmul(a22, b21 - b11, mod)
    m5 = mmul(a11 + a12, b22, mod)
    m6 = mmul(a21 - a11, b11 + b12, mod)
    m7 = mmul(a12 - a22, b21 + b22, mod)
    c11 = m1 + m4 - m5 + m7
    c12 = m3 + m5
    c21 = m2 + m4
    c22 = m1 - m2 + m3 + m6
    result = np.block([[c11, c12], [c21, c22]])
    return result % mod


def __complete_to_power(x, base):
    value = 1
    while value < x:
        value *= base
    return value


def __complete_matrix(mat, size):
    # print('Completing ', mat)
    # assert(mat.shape[0] <= size[0])
    # assert(mat.shape[1] <= size[1])
    return np.pad(
        mat,
        ((0, size[0] - mat.shape[0]), (0, size[1] - mat.shape[1])),
        'constant',
        constant_values=0)


def __disassemble_matrix(m):
    # print('Disassembling', m)
    # assert(m.shape[0] % 2 == 0)
    # assert(m.shape[1] % 2 == 0)

    end_row = m.shape[0]
    end_col = m.shape[1]
    split_row = end_row // 2
    split_col = end_col // 2
    return (m[0:split_row, 0:split_col],
            m[0:split_row, split_col:end_col],
            m[split_row:end_row, 0:split_col],
            m[split_row:end_row, split_col:end_col])


matrix = read_matrix()
result = log_power(matrix, matrix.shape[0], multiply_matrices)
print_matrix(result)
