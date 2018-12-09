import sys
import numpy as np

# Constants


VAR = 0
NOT = -1
OR = 1
AND = 2

WELL_KNOWN_CYCLE_A = np.array([2, 3, 4, 5, 1]) - 1
WELL_KNOWN_CYCLE_B = np.array([3, 5, 2, 1, 4]) - 1
IDENTITY_CYCLE = np.arange(5)


# Algorithm


def replace_disjunction(circuit):
    n = len(circuit)
    current = n
    for i in range(n):
        node = circuit[i]
        if node[0] == OR:
            in1 = node[1]
            in2 = node[2]

            circuit.append((NOT, in1))
            not_in1 = current
            current += 1

            circuit.append((NOT, in2))
            not_in2 = current
            current += 1

            circuit.append((AND, not_in1, not_in2))
            both_false = current
            current += 1

            circuit[i] = (NOT, both_false)

    return circuit

# pbp := ((cycle), [instruction])
# instruction := (variable id, cycle on true, cycle on false)
# k-cycle = np.array() of length k
# permutation - 2 arrays

def apply_cycle(cycle, values):
    current = values[cycle[-1]]
    for k in cycle:
        new_ = values[k]
        values[k] = current
        current = new_
    return values


def apply_permutation(from_, to_, values):
    values[to_] = values[from_]
    return values


def apply_permutation(to, values):
    values[to] = values
    return values

def compose(perm1, perm2):
    return np.arange(len(perm1))[perm2[perm1]]

def compose(*argv):
    if len(argv) == 0:
        return None
    result = argv[0]
    for perm in argv[1:]:
        result = compose(result, perm) # todo: there could be an error with order of composition
    return result

def revert(perm):
    return np.argsort(perm)


# ccc = np.array([3, 1, 2, 4]) - 1
# print(revert(ccc) + 1)
# exit()


# from_ = np.array([3,2,1]) - 1
# to_ = np.array([2, 1, 3]) - 1
# values = np.array(['un', 'deux', 'trois'])
# print(apply_permutation(from_, to_, values))
# exit()

# a = np.array([3,1,2]) - 1
# bb = np.array([2, 1, 3]) - 1
# print(bb[np.argsort(a)])

def change_permutation(pbp, new_cycle):
    print(pbp)
    print(new_cycle)
    new_pbp = pbp
    new_pbp[0] = new_cycle
    gamma = new_cycle[np.argsort(pbp[0])]
    gamma_rev = revert(gamma)
    print(gamma_rev)
    new_pbp[1][0] = (new_pbp[1][0][0], compose(gamma, pbp[1][0][1]), compose(gamma, pbp[1][0][2]))
    new_pbp[1][-1] = (new_pbp[1][-1][0], compose(pbp[1][-1][1], gamma_rev), compose(pbp[1][-1][2], gamma_rev))
    print(gamma)
    print(pbp)
    print(new_pbp)


def build_var(var):
    print(var)
    new_pbp = (WELL_KNOWN_CYCLE_A, [(var, WELL_KNOWN_CYCLE_A, IDENTITY_CYCLE)])
    print(new_pbp)
    return new_pbp


def build_not(pbp):
    print(pbp)
    sigma = pbp[0]
    sigma_rev = revert(sigma)
    new_pbp = change_permutation(pbp, sigma_rev)
    new_pbp[0] = sigma_rev
    new_pbp[1][-1] = (new_pbp[1][-1][0], \
        compose(pbp[1][-1][1], sigma_rev),\
        compose(pbp[1][-1][2], sigma_rev))
    print(pbp)
    print(new_pbp)
    return new_pbp


def build_and(pbp1, pbp2):
    print(pbp1)
    print(pbp2)
    sigma = WELL_KNOWN_CYCLE_A
    tau = WELL_KNOWN_CYCLE_B
    pbp1_new = change_permutation(pbp1, sigma)
    pbp2_new = change_permutation(pbp2, tau)

    print(pbp1_new)
    print(pbp2_new)

    sigma_rev = revert(sigma)
    tau_rev = revert(sigma)
    pbp1_rev = change_permutation(pbp1_new, sigma_rev)
    pbp2_rev = change_permutation(pbp2_new, tau_rev)

    new_perm = compose(sigma, tau, sigma_rev, tau_rev)
    new_commands = pbp1_new[1] + pbp2_new[1] + pbp1_rev[1] + pbp2_rev[1]
    print(new_perm)
    print(new_commands)

    return (new_perm, new_commands)


# Tests




# Application


def get_input():
    _ = int(input())
    tokens = [line.split() for line in sys.stdin]
    circuit = []
    for t in tokens:
        type_ = t[0]
        if type_[0] == 'V':
            name = t[1]
            circuit.append((VAR, name))
        elif type_[0] == 'N':
            in_ = int(t[1])
            circuit.append((NOT, in_))
        elif type_[0] == 'A':
            in_1 = int(t[1])
            in_2 = int(t[2])
            circuit.append((AND, in_1, in_2))
        elif type_[0] == 'O':
            in_1 = int(t[1])
            in_2 = int(t[2])
            circuit.append((OR, in_1, in_2))
    return circuit


def run_solution():
    circuit = get_input()
    print(circuit)
    circuit = replace_disjunction(circuit)
    print(circuit)


run_solution()
