from itertools import product, combinations
from .utils import *
from .codes_q import *
from .NHow import *

triple_intersect = lambda x, y, z: np.logical_and(np.logical_and(x,y),z)
set_of_len_l_tuples = lambda n_max, l: set(tuple(sorted(s)) for s in product(range(n_max), repeat=l) if len(set(s)) == l)
logical_and_list = lambda mat: reduce(np.logical_and, mat)

def solve_for_metric(log, stab):
    n = log.shape[1]
    constraints = []
    for stab1 in stab:
        for stab2 in stab:
            for stab3 in stab:
                constraints.append(triple_intersect(stab1, stab2, stab3))
            for log1 in log:
                constraints.append(triple_intersect(log1, stab1, stab2))
        for log1 in log:
            for log2 in log:
                constraints.append(triple_intersect(log1, log2, stab1))
    constraints = np.array(constraints).astype(int)
    # print(format_np_array_compact(constraints))
    return kernel(constraints)[0]

def possible_logical_magic_gates(log, metric, X_basis=False):
    # still under construction
    output = "" # only print if there is something interesting
    output += f"metric: {metric}\n"
    if X_basis: output += "X basis, so interpret all gates as Hadamard conjugate\n"
    k, _ = log.shape
    has_magic = False
    for i in range(k):
        for j in range(i, k):
            for l in range(j, k):
                intersection = np.logical_and(metric, triple_intersect(log[i], log[j], log[l])).astype(int).sum()
                if intersection % 2:
                    has_magic = True
                    if i==j==l: output += f"T on logical {i}\n"
                    if i==j and j!=l: output += f"CS on logical {i}, {l}\n"
                    if len(set([i,j,l]))==3: output += f"CCZ on logical {i}, {j}, {l}"
    if has_magic: print(output)

def parse_gate(coeff, num_term, N): # return gate name and level
    if N == 4:
        if num_term == 1:
            if coeff == 1: return "S", 2
            if coeff == 2: return "Z", 1
            if coeff == 3: return "S3", 2 
        if num_term == 2:
            if coeff == 2: return "CZ", 2
    if N == 8:   
        if num_term == 1:
            if coeff % 2 == 0: return parse_gate(coeff//2, num_term, N//2)
            return f"T{coeff}", 3
        if num_term == 2:
            if coeff == 4: return "CZ", 2
            if coeff == 2: return "CS", 3
            if coeff == 6: return "CS3", 3 
        if num_term == 3:
            if coeff == 4: return "CCZ", 3
    if N == 16:
        if num_term == 1:
            if coeff % 2 == 0: return parse_gate(coeff//2, num_term, N//2)
            return f"√T{coeff}", 4
        if num_term == 2:
            if coeff % 4 == 0: return parse_gate(coeff//2, num_term, N//2)
            return f"CT{coeff//2}", 4
        if num_term == 3:
            if coeff % 8 == 0: return parse_gate(coeff//2, num_term, N//2)
            return f"CCS{coeff//4}", 4
        if num_term == 4:
            if coeff == 8: return "CCCZ", 4
    print(f"coeff={coeff}, #terms={num_term}, modulo {N} not a supported gate for the phase polynomial")
    return False

def transversal_Z_rotation(log, stab, t, parse=True, verbose=True): # turn parse to False for t>4
    # construct a matrix with entries in Z_{2^t}
    # kernel is a metric, where entry i is to be interpreted as applying Z(1/2^t)^{metric[i]} to physical qubit i
    # 2^{l-1} * wt_metric(g^{a_1} ^ ... ^ g^{a_l}) = 0 (mod 2^t),  for every l <= t
    # where (a_1, ..., a_l) \in [1,...,m]^{\times l} \ [1,...,k]^{\times l}
    # make sure Lx (log), Sx (stab) are full-rank
    k, n = log.shape
    num_stab = stab.shape[0]
    m = k + num_stab
    N = 2**t
    mat = np.vstack((log, stab))
    all_rows = []
    for l in range(1, t+1):
        weight = 2**(l-1)
        for a in set_of_len_l_tuples(m, l) - set_of_len_l_tuples(k, l):
            all_rows.append(logical_and_list([mat[i] for i in a]).astype(int) * weight)
    all_rows = ZMat(all_rows, N)
    # print("constraint matrix\n", all_rows)
    metric = getK(all_rows, N)
    # print(f"level{t}, metric{metric}")
    residue = metric.astype(int) % 2
    if not residue.any():
        return False, -1, None
    maximal_level = 1
    has_magic = False
    all_magic_generator_str = ""
    for metric in metric[residue.any(axis=1)]:
        logical_phase_poly = ""
        logical_gate_str = ""
        level_3_logical_str = ""
        metric_maximal_level = 1
        for l in range(1, t+1):
            weight = 2**(l-1) * np.ones((n), dtype=int)
            weight = metric * weight
            for a in set_of_len_l_tuples(k, l):
                intersection = logical_and_list([log[i] for i in a])
                coeff = (-1)**(l-1) * weight[intersection.astype(bool)].sum() % N
                if coeff == 0: continue
                logical_phase_poly += f"{coeff}*"
                for i in a: logical_phase_poly += f"x{i}"
                logical_phase_poly += " + "
                if parse:
                    gate_name, level = parse_gate(coeff, len(a), N)
                    maximal_level = max(maximal_level, level)
                    metric_maximal_level = max(metric_maximal_level, level)
                    logical_gate_str += gate_name + f"[{",".join([str(i) for i in a])}] "
                    level_3_logical_str += gate_name + f"[{",".join([str(i) for i in a])}] "
        
        if len(logical_phase_poly) > 0:
            if metric_maximal_level >=3: has_magic = True
            if verbose and metric_maximal_level >= 3:
                all_magic_generator_str += f"Physical gate, Z rotation Z({N}) of power {metric} on each qubit\n"
                logical_phase_poly = logical_phase_poly[:-3] + f" (mod {N})"
                all_magic_generator_str += f"Logical phase polynomial: {logical_phase_poly}\n"
                all_magic_generator_str += f"Logical gate: {logical_gate_str}\n"
            if not verbose and metric_maximal_level >= 3: # return immediately, do not bother getting all the magic generators
                return True, metric_maximal_level, (f"Z({N}){metric}", level_3_logical_str)
    return has_magic, maximal_level, all_magic_generator_str

def transversal_Z_rotation_all(log, stab, t, parse=True, verbose=True): # turn parse to False for t>4
    # construct a matrix with entries in Z_{2^t}
    # kernel is a metric, where entry i is to be interpreted as applying Z(1/2^t)^{metric[i]} to physical qubit i
    # 2^{l-1} * wt_metric(g^{a_1} ^ ... ^ g^{a_l}) = 0 (mod 2^t),  for every l <= t
    # where (a_1, ..., a_l) \in [1,...,m]^{\times l} \ [1,...,k]^{\times l}
    # make sure Lx (log), Sx (stab) are full-rank
    k, n = log.shape
    num_stab = stab.shape[0]
    m = k + num_stab
    N = 2**t
    mat = np.vstack((log, stab))
    all_rows = []
    for l in range(1, t+1):
        weight = 2**(l-1)
        for a in set_of_len_l_tuples(m, l) - set_of_len_l_tuples(k, l):
            all_rows.append(logical_and_list([mat[i] for i in a]).astype(int) * weight)
    all_rows = ZMat(all_rows, N)
    # print("constraint matrix\n", all_rows)
    metric = getK(all_rows, N)
    # print(f"level{t}, metric{metric}")
    residue = metric.astype(int) % 2
    if not residue.any():
        return False, -1, None
    maximal_level = 1
    has_magic = False
    all_magic_generator_str = ""
    str_return = []
    metric_return = []
    for metric in metric[residue.any(axis=1)]:
        logical_phase_poly = ""
        logical_gate_str = ""
        level_3_logical_str = ""
        metric_maximal_level = 1
        for l in range(1, t+1):
            weight = 2**(l-1) * np.ones((n), dtype=int)
            weight = metric * weight
            for a in set_of_len_l_tuples(k, l):
                intersection = logical_and_list([log[i] for i in a])
                coeff = (-1)**(l-1) * weight[intersection.astype(bool)].sum() % N
                if coeff == 0: continue
                logical_phase_poly += f"{coeff}*"
                for i in a: logical_phase_poly += f"x{i}"
                logical_phase_poly += " + "
                if parse:
                    gate_name, level = parse_gate(coeff, len(a), N)
                    maximal_level = max(maximal_level, level)
                    metric_maximal_level = max(metric_maximal_level, level)
                    logical_gate_str += gate_name + f"[{",".join([str(i) for i in a])}] "
                    level_3_logical_str += gate_name + f"[{",".join([str(i) for i in a])}] "
        
        if len(logical_phase_poly) > 0:
            all_magic_generator_str_temp = f"Physical gate, Z rotation Z({N}) of power {metric} on each qubit\n"
            logical_phase_poly = logical_phase_poly[:-3] + f" (mod {N})"
            all_magic_generator_str_temp += f"Logical phase polynomial: {logical_phase_poly}\n"
            all_magic_generator_str_temp += f"Logical gate: {logical_gate_str}\n"
            str_return.append(all_magic_generator_str_temp)
            metric_return.append(metric)

    return str_return, np.array(metric_return)

def metric_to_phys_impl(metric, V, n):
    S_rot = np.zeros(n, dtype=np.int16)
    CZs = []
    assert len(metric) == V.shape[0]
    for idx, V_row in enumerate(V):
        if sum(V_row) == 1: # original qubit
            qubit_idx = np.where(V_row)[0][0]
            S_rot[qubit_idx] = metric[idx]
        elif sum(V_row) == 2: # gadget qubit
            i,j = sorted(np.where(V_row)[0]) # added on i, j
            if metric[idx] == 0:
                continue
            elif metric[idx] == 1: # S gate -> S[i] S[j] CZ[i,j]
                # XOR the CZ[i,j]
                if (i,j) in CZs: CZs.remove((i,j))
                else: CZs.append((i,j))
                S_rot[i] += 1
                S_rot[j] += 1
            elif metric[idx] == 2: # Z gate -> Z[i] Z[j]
                S_rot[i] += 2
                S_rot[j] += 2
            elif metric[idx] == 3: # Sdagger -> Sdagger[i] Sdagger[j] CZ[i,j]
                if (i,j) in CZs: CZs.remove((i,j))
                else: CZs.append((i,j))
                S_rot[i] += 3
                S_rot[j] += 3

    phys_impl_str = ""
    for idx, rot in enumerate(S_rot):
        if rot % 4 == 0: continue
        if rot % 2 == 0: 
            phys_impl_str += f"Z[{idx}] "
        else:
            phys_impl_str += f"S{rot%4}[{idx}] "
    for (i,j) in CZs:
        phys_impl_str += f"CZ[{i},{j}]"
    return phys_impl_str

############# from CSSLO repo XCP_algebra.py #############
def MntSubsets(S,t,mink=1):
    '''Return list of subsets of max weight t whose support is a subset of S'''
    return {s for k in range(mink, t+1) for s in combinations(S,k)}

def Mnt_partition(cList,n,t,mink=1):
    '''Return a set of binary vectors of length n and weight between mink and t
    including subsets of cycles in cList. cList is a permutation in cycle form. '''
    temp = set()
    for c in cList:
        temp.update(MntSubsets(c,t,mink))
    temp = [(n-len(s),tuple(set2Bin(n,s))) for s in temp]
    temp = sorted(temp,reverse=True)
    return ZMat([s[1] for s in temp])
###########################################################

def fold_diagonal_gate(Lx, Hx, cList=None):
    '''
    Under contruction.
    cList for pairs of qubits to add the gadget on
    '''
    r,n = np.shape(Hx)
    k,n = np.shape(Lx)

    if cList is None:
        ## All binary vectors of length n weight 1-t
        V = Mnt(n, 2)
    else:
        ## V based on partition supplied to algorithm
        V = Mnt_partition(cList, n, 2)

    print("V", V)
    print("V shape", V.shape)
    print(V.sum(axis=1))
    ## Construct Embedded Code
    Hx_embed = Hx @ V.T % 2
    Lx_embed = Lx @ V.T % 2
    ## Find diagonal logical operators at level t
    # print(transversal_Z_rotation_all(LX_V, SX_V, 2))
    str_all, metric_all = transversal_Z_rotation_all(Lx_embed, Hx_embed, 2)
    print("number of metric", len(str_all))
    for str, metric in zip(str_all, metric_all):
        print("Physical string", metric_to_phys_impl(metric, V, n))
        print(str)