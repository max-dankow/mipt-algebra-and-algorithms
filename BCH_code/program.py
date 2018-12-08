import numpy as np

# Algorithm


def build_hadamar_matrix_paley(n):
    p = n - 1
    # assert n = p + 1, where p is prime and n mod 4 = 0

    residues = -np.ones(2 * p, dtype=int)
    squares = (np.arange(1, (p - 1) // 2 + 1, dtype=int) ** 2) % p
    residues[squares] = 1
    residues[squares + p] = 1

    result = np.ones(shape=(n, n), dtype=int)
    for i in range(1, n):
        for j in range(1, n):
            result[i, j] = residues[j - i + p]

    return result


# Tests


def test_hadamar_def(mat):
    epsilon = 1e-7
    n = mat.shape[0]
    assert(mat.shape[1] == n)
    diff = np.matmul(mat, mat.T) - n * np.identity(mat.shape[0])
    assert np.all(diff < epsilon), 'Hadamar matrix check failed'


def test_codes(codes):
    n = codes.shape[1]
    for i in range(2 * n):
        for j in range(2 * n):
            if i == j:
                continue
            dist = np.sum(codes[i] != codes[j])
            assert dist >= (n // 2),\
                'Codes distance %s is less than n//2' % dist


def test_solution():
    n = 12
    codes = compute_codes(n)
    test_codes(codes)


# Application


def get_input():
    return int(input())


def compute_codes(n):
    hadamar_mat = build_hadamar_matrix_paley(n)

    codes_positive = hadamar_mat
    codes_positive[codes_positive == -1] = 0
    codes_negative = 1 - codes_positive
    codes = np.vstack((codes_positive, codes_negative))
    return codes


def print_codes(codes):
    output = '\n'.join([''.join(str(x) for x in code) for code in codes])
    print(output)


def run_solution():
    n = get_input()
    codes = compute_codes(n)
    print_codes(codes)


# test_solution()
run_solution()
