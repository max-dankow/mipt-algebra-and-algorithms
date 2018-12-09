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


def compose(*argv):
    if len(argv) == 0:
        return None
    result = argv[0]
    for perm in argv[1:]:
        result = np.arange(len(result))[perm[result]]
        # todo: there could be an error with order of composition
    return result

def revert(perm):
    return np.argsort(perm)

def cycle_to_chain(cycle):
    k = 0
    result = []
    for _ in cycle:
        result.append(cycle[k])
        k = cycle[k]
    assert k == 0, 'Cycle is not actually cyclic'
    # print('Cycle', cycle,'to chain', result)
    return np.array(result)

def change_permutation(pbp, new_cycle):
    # print('original before transform', pbp)
    # print('new cycle', new_cycle)
    gamma = cycle_to_chain(new_cycle)[np.argsort(cycle_to_chain(pbp[0]))]
    gamma_rev = revert(gamma)
    # print('gamma', gamma)
    # print('gamma_rev', gamma_rev)

    new_commands = pbp[1].copy()
    new_commands[0] = (new_commands[0][0],
        compose(gamma, new_commands[0][1]),\
        compose(gamma, new_commands[0][2]))

    new_commands[-1] = (new_commands[-1][0],
        compose(new_commands[-1][1], gamma_rev),\
        compose(new_commands[-1][2], gamma_rev))

    new_pbp = (new_cycle, new_commands)

    # print('original after transform', pbp)
    # print('new after transform', new_pbp)
    return new_pbp

def build_var(var):
    # print(var)
    new_pbp = (WELL_KNOWN_CYCLE_A, [(var, WELL_KNOWN_CYCLE_A, IDENTITY_CYCLE)])
    # print(new_pbp)
    return new_pbp


def build_not(pbp):
    # print('original before not', pbp)
    sigma = pbp[0]
    sigma_rev = revert(sigma)
    new_pbp = change_permutation(pbp, sigma_rev)
    new_pbp[1][-1] = (new_pbp[1][-1][0], \
        compose(pbp[1][-1][1], sigma_rev),\
        compose(pbp[1][-1][2], sigma_rev))
    # print('original after not', pbp)
    # print('new after not', new_pbp)
    return new_pbp


def build_and(pbp1, pbp2):
    # print(pbp1)
    # print(pbp2)
    sigma = WELL_KNOWN_CYCLE_A
    tau = WELL_KNOWN_CYCLE_B
    pbp1_new = change_permutation(pbp1, sigma)
    pbp2_new = change_permutation(pbp2, tau)

    # print(pbp1_new)
    # print(pbp2_new)

    sigma_rev = revert(sigma)
    tau_rev = revert(tau)
    pbp1_rev = change_permutation(pbp1_new, sigma_rev)
    pbp2_rev = change_permutation(pbp2_new, tau_rev)

    new_perm = compose(sigma, tau, sigma_rev, tau_rev)
    new_commands = pbp1_new[1] + pbp2_new[1] + pbp1_rev[1] + pbp2_rev[1]
    # print(new_perm)
    # print(new_commands)

    return (new_perm, new_commands)


def build_rec(circuit, current, pbps):
    # Caller should check if result is already calculated
    # print(current)
    type_ = circuit[current][0]
    if type_ == VAR:
        pbps[current] = build_var(current)
    elif type_ == NOT:
        in_ = circuit[current][1]
        if pbps[in_] is None:
            build_rec(circuit, in_, pbps)
        pbps[current] = build_not(pbps[in_])
    elif type_ == AND:
        in1 = circuit[current][1]
        in2 = circuit[current][2]
        if pbps[in1] is None:
            build_rec(circuit, in1, pbps)
        if pbps[in2] is None:
            build_rec(circuit, in2, pbps)
        pbps[current] = build_and(pbps[in1], pbps[in2])
    else:
        assert False, 'Circuit contains OR, but it shouldn''t'

def buid_pbp(circuit, output_node):
    current = 0
    pbps = np.empty(shape=(len(circuit),), dtype=tuple)
    # print(pbps)

    for _ in range(len(circuit)):
        if pbps[current] is None:
            build_rec(circuit, current, pbps)
        current += 1
        # print(pbps)


    return pbps[output_node]

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


def print_result(pbp, circuit):
    k = 5
    for i, cmd in enumerate(pbp[1]):
        var_name = circuit[cmd[0]][1]
        alpha = cmd[1]
        beta = cmd[2]
        for t in range(len(cmd[1])):
            print(var_name, k + alpha[t], k + beta[t])
        k += 5
    output = np.array([False, True, True, True, True])
    for x in output:
        print(x)

def run_solution():
    circuit = get_input()
    output_node = len(circuit) - 1
    # print(circuit)
    circuit = replace_disjunction(circuit)
    # print(circuit)
    result = buid_pbp(circuit, output_node)
    # print('RESULT', result)
    print_result(result, circuit)


run_solution()
