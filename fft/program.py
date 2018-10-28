import numpy as np
import time
from math import pi, sin, cos


def get_input():
    return list(map(float, input().strip().split()))


def print_result(result):
    print(' '.join(f'{x.real},{x.imag}' for x in result))


def solve_fft(coeffs):
    coeffs = complete_coeffs(coeffs)
    result = fft(coeffs, len(coeffs))
    return result


def complete_coeffs(coeffs):
    power_2 = 1
    while (1 << power_2) < len(coeffs):
        power_2 += 1

    if 1 << power_2 != len(coeffs):
        return [0] * ((1 << power_2) - len(coeffs)) + coeffs
    else:
        return coeffs


def fft(coeffs, n, state=(1, 0)):
    assert(len(coeffs) % 2 == 0)
    half_n = n // 2

    base_power, base = state
    if base < len(coeffs) and base + base_power >= len(coeffs):
        return [coeffs[base]]

    a_0_state = fix_next_even(state)
    fft_0 = fft(coeffs, half_n, a_0_state)

    a_1_state = fix_next_odd(state)
    fft_1 = fft(coeffs, half_n, a_1_state)

    result = np.empty(shape=n, dtype=complex)

    for i in range(half_n):
        w_i = get_w_k(n, i)
        result[i] = fft_0[i] + w_i * fft_1[i]
        result[i + half_n] = fft_0[i] - w_i * fft_1[i]

    return result


def gen(state, max_):
    base_power, base = state
    x = base
    while(x <= max_):
        yield x
        x += base_power


def fix_next_odd(state):
    base_power, base = state
    return (base_power << 1, base + base_power)


def fix_next_even(state):
    base_power, base = state
    return (base_power << 1, base)


def get_w_k(n, k):
    return cos(2 * pi * k / n) + 1j * sin(2 * pi * k / n)


def solve_base(coeffs):
    n = len(coeffs)
    return [
        polynomial(coeffs, get_w_k(n, k))
        for k in range(n)
    ]


def polynomial(coeffs, x):
    return sum(a * x ** k for k, a in enumerate(coeffs))


def test(solve, n, max_power):
    for _ in range(n):
        d = 1 << np.random.randint(1, max_power)
        coeffs = np.random.random(d)
        assert_correct(coeffs, solve)


def assert_correct(coeffs, solve, epsilon=1e-6):
    expected = solve_base(coeffs)

    actual = solve(coeffs)
    assert(len(expected) == len(actual))
    for i in range(len(expected)):
        assert(expected[i] - actual[i] < epsilon)


def perf():
    times = []
    n = 1000
    for _ in range(n):
        coeffs = np.random.random(1 << 10)
        time_s = time.time()

        solve_fft(coeffs)
        times.append(time.time() - time_s)
    return sum(times) / len(times)

# test(solve_fft, 100, 7)
# print(perf())

coeffs = get_input()
result = solve_fft(coeffs)

print_result(result)
