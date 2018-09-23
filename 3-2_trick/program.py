import sys

def next():
    global v
    v += 1
    return v

def and_(a, b):
    next()
    print("GATE %d AND %d %d" % (v, a, b))
    return v

def or_(a, b):
    next()
    print("GATE %d OR %d %d" % (v, a, b))
    return v

def not_(a):
    next()
    print("GATE %d NOT %d" % (v, a))
    return v

def generate_one(i):
    a = i
    not_a = not_(a)  # получается что, первая сгенерированная вершина - отрицание 0-ой
    b = i + n
    c = i + 2 * n
    b_and_c = and_(b, c)
    b_or_c = or_(b, c)
    b_xor_c = and_(not_(b_and_c), b_or_c)
    b_eq_c = or_(not_(b_or_c), b_and_c)
    result_x = or_(and_(not_a, b_xor_c), and_(a, b_eq_c))

    result_y = or_(and_(a, b_or_c), b_and_c)

    return (result_x, result_y)


n = int(sys.stdin.readline())
v = 3*n - 1  # last input

for i in range(n):
    x, y = generate_one(i)
    print("OUTPUT %d %d" % (i, x))
    print("OUTPUT %d %d" % (i + n + 2, y))

# для экономии возьмем a_0 и not(a_0), поскольку отрицание будет и так сгенерировано
zero = and_(0, 3*n)

print("OUTPUT %d %d" % (n, zero))
print("OUTPUT %d %d" % (n + 1, zero))