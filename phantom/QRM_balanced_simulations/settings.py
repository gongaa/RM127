import stim
import numpy as np
from itertools import combinations, product
import sys
sys.path.append("../")
from src.codes_q import css_code, kernel
####################### Settings ######################
global m, N, l, K, d
m = 6 # keep it the same as in structure_test.py
N = 2 ** m 
l = 3 # choose a number between [1,m-1]
K = m-l+1
d = min(2**(m-l), 2**l)
# phantom QRM parameter [[2^m, m-l+1, 2^{m-l} / 2^l]]
bin_wt = lambda i: bin(i)[2:].count('1')
bit_rev = lambda t: int(bin(t)[2:].rjust(m, '0')[::-1], 2)
int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(m, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)
def Eij(i,j):
    mat = np.eye(m, dtype=int)
    mat[i,j] = 1
    return mat

# current error rate Sasha gave
# p_CNOT = 3e-3
# p_single_qubit_gate = 2e-4
# p_meas = 5e-3
# p_reset = 3e-3
# p_idle = 1e-5
# p_idle_perm = 5e-4
######################### error rate settings ########################
p_CNOT = 0.003
p_single_qubit_gate = p_CNOT / 15.0
p_meas = 5.0 * p_CNOT / 3.0
p_reset = p_CNOT
p_idle = p_CNOT / 300.0
p_idle_perm = p_CNOT / 6.0
######################################################################

logical_indices_binary = ["000111", "001011", "010011", "100011"]
logical_indices = [N-1-bin2int(idx) for idx in logical_indices_binary]
print("logical indices", logical_indices)
extra_X_stab_indices_binary = ["110010", "101010", "100110", "011010", "010110", "001110", "111000", "110100", "101100", "011100"]
extra_X_stab_indices = [N-1-bin2int(idx) for idx in extra_X_stab_indices_binary]
print("extra X stabilizer indices", extra_X_stab_indices)

F = np.array([[1,0],[1,1]])
E = F
for i in range(m-1):
    E = np.kron(E, F)
Lx = E[logical_indices]
Lz = E.T[logical_indices]
# X stabilizer generators: all degree < l monomials
X_stab_indices = [N-1-i for i in range(N) if bin_wt(i) < l]
X_stab_indices += extra_X_stab_indices
Hx = E[X_stab_indices]
mat = np.vstack((Lx, Hx))
Hz = kernel(mat)[0]
print("X stabilizer shape", Hx.shape)
print("Z stabilizer shape", Hz.shape)
code = css_code(Hx, Hz)
print(f"code parameter [[{code.N},{code.K}]]")

# for fold-S
permutation = [i for i in range(N)]
for i in range(N):
    i_bin = int2bin(N-1-i)
    i_bin = [1-i_bin[5], 1-i_bin[4], 1-i_bin[3], 1-i_bin[2], 1-i_bin[1], 1-i_bin[0]]
    permutation[i] = N-1-bin2int(i_bin)
fixed_point = [i for (idx, i) in enumerate(permutation) if i==idx]
print("fixed points", fixed_point)
involution = []
for i in range(N):
    j = permutation[i]
    if i != j:
        assert permutation[j] == i, "Not an involution!"
        temp = sorted([i,j])
        if temp not in involution:
            involution.append(temp)
    else:
        involution.append([i])


def append_hypercube_encoding_circuit(circuit, offset, permute=None, p_CNOT=0.0): # initialization is not included
    if permute is None:
        permute = [i for i in range(N)]
    for r in range(m):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N:
                    circuit.append("CNOT", [offset + permute[j+i+sep], offset + permute[j+i]])
                    if p_CNOT > 0.0:
                        circuit.append("DEPOLARIZE2", [offset + permute[j+i+sep], offset + permute[j+i]], p_CNOT)

def append_initialization(circuit, state, offset, permute=None, p_reset=0.0):
    assert len(state) == K, "initial state length mismatch with K !"
    for i in range(K):
        assert state[i] in ["0","1","+","-"], "invalid state in initialization !"

    if permute is None:
        permute = [i for i in range(N)]
    
    for i in range(N):
        if bin_wt(N-1-i) < l or (i in extra_X_stab_indices): # |+>
            circuit.append("RX", offset + permute[i])
            if p_reset > 0.0:
                circuit.append("Z_ERROR", offset + permute[i], p_reset)
        elif i not in logical_indices: # |0>
            circuit.append("R", offset + permute[i])
            if p_reset > 0.0:
                circuit.append("X_ERROR", offset + permute[i], p_reset)
    for i in range(K):
        log_ind = logical_indices[i]
        if state[i] == "0" or state[i] == "1":
            circuit.append("R", offset + permute[log_ind])
            if p_reset > 0.0:
                circuit.append("X_ERROR", offset + permute[log_ind], p_reset)
            if state[i] == "1":
                circuit.append("X", offset + permute[log_ind])
        elif state[i] == "+" or state[i] == "-":
            circuit.append("RX", offset + permute[log_ind])
            if p_reset > 0.0:
                circuit.append("Z_ERROR", offset + permute[log_ind], p_reset)
            if state[i] == "-":
                circuit.append("Z", offset + permute[log_ind])

def append_measurement(circuit, state, offset, detector=False):
    assert len(state) == K, "initial state length mismatch with K !"
    for i in range(K):
        assert state[i] in ["0","1","+","-"], "invalid state in measurement!"
    
    for i in range(N):
        if bin_wt(N-1-i) < l or (i in extra_X_stab_indices): # |+>
            circuit.append("MX", offset + i)
        elif i not in logical_indices: # |0>
            circuit.append("M", offset + i)
    for i in range(K):
        log_ind = logical_indices[i]
        if state[i] == "0" or state[i] == "1":
            if state[i] == "1":
                circuit.append("X", offset + log_ind)
            circuit.append("M", offset + log_ind)
        elif state[i] == "+" or state[i] == "-":
            if state[i] == "-":
                circuit.append("Z", offset + log_ind)
            circuit.append("MX", offset + log_ind)

    if detector:
        for i in range(K):
            circuit.append("DETECTOR", stim.target_rec(-K+i))



# create weight-16 stabilizers for QRM
# a canonical choice of minimum-weight basis of RM(r,m):
# for every T \subseteq [m] with |T| <= r. take the produce of x_i for i \in T and 1+x_j for the first r-|T| indices j \notin T
def canonical_min_weight_basis(r: int, m: int):
    """
    Return a list of canonical minimum-weight generators for RM(r,m).
    Indices are 1-based in T and N for readability; internal vectors use 0/1 ints.
    """
    if not (0 <= r <= m):
        raise ValueError("Require 0 <= r <= m.")
    
    # All T with |T| <= r, ordered first by size then lex
    Ts = []
    for tsize in range(r + 1):
        Ts.extend(combinations(range(1, m + 1), tsize))  # 1-based indices

    basis = []
    for T in Ts:
        Tset = set(T)
        need = r - len(T)
        # pool = indices not in T, in 1..m
        pool = [i for i in range(1, m + 1) if i not in Tset]

        # pick N_T: first use indices > max(T), then wrap
        greater_than = []
        if len(T) > 0:
            tmax = max(T)
            greater_than = [i for i in pool if i > tmax]
        else:
            tmax = 0
            greater_than = pool[:]  # everything is "greater" than 0

        N = []
        # take as many as possible from greater_than (in increasing order)
        N.extend(greater_than[:need])
        # if not enough, wrap to the smallest available (in increasing order)
        if len(N) < need:
            remaining = [i for i in pool if i not in N]
            N.extend(remaining[: (need - len(N))])

        N = tuple(N)

        # Build human-readable ANF string
        def term_x(i): return f"x{i}"
        def term_1px(j): return f"(1+x{j})"
        anf_parts = [term_x(i) for i in T] + [term_1px(j) for j in N]
        anf = " * ".join(anf_parts) if anf_parts else "1"

        # Evaluate the word on {0,1}^m in lex order:
        # points are tuples (x1,...,xm) with x1 changing slowest (MSB)
        word = []
        fixed1 = set(T)
        fixed0 = set(N)
        for x in product((0, 1), repeat=m):
            ok = True
            for i in fixed1:
                if x[i - 1] != 1:
                    ok = False
                    break
            if ok:
                for j in fixed0:
                    if x[j - 1] != 0:
                        ok = False
                        break
            word.append(1 if ok else 0)

        # Sanity checks (can comment out for speed)
        wt = sum(word)
        expected = 2 ** (m - r)
        if wt != expected:
            raise RuntimeError(f"Weight mismatch for T={T}, N={N}: got {wt}, expected {expected}")

        basis.append({
            "T": T,
            "N": N,
            "anf": anf,
            "word": word,  # length 2^m list of 0/1
        })

    return basis


wt_8_Z_stab_indices_binary = []
for idx in [int2bin(i) for i in range(N) if bin_wt(i) == l]:
    idx = ''.join(map(str, idx))
    if idx not in logical_indices_binary + extra_X_stab_indices_binary:
        wt_8_Z_stab_indices_binary.append(idx)
print(wt_8_Z_stab_indices_binary)
wt_8_Z_stab_indices = [N-1-bin2int(idx) for idx in wt_8_Z_stab_indices_binary]
wt_8_Hx = E[extra_X_stab_indices]
wt_8_Hz = E.T[wt_8_Z_stab_indices]


stab = []
r = l-1
basis = canonical_min_weight_basis(r, m)
print(f"RM({r},{m}) canonical min-weight basis size = {len(basis)}")
# Print a few generators
for b in basis:
    # print(f"T={b['T']}, N={b['N']}, anf={b['anf']}, weight={sum(b['word'])}")
    stab.append(b['word'])
stab = np.array(stab)

low_wt_Hx = np.vstack((wt_8_Hx, stab[:,::-1])) # bit-reversed is the same code, for aesthetic reason
low_wt_Hz = np.vstack((wt_8_Hz, stab))
assert not (Hx @ Hz.T % 2).any()