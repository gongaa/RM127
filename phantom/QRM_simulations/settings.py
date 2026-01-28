import stim
import numpy as np
import sys
sys.path.append("../")
from QRM_balanced_simulations.settings import canonical_min_weight_basis
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

p_CNOT = 3e-3
p_single_qubit_gate = 2e-4
p_reset = 3e-3
p_meas = 5e-3
p_idle = 1e-5
p_permutation_idle = 5e-4

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
    logical_indices = []
    for i in range(l-1, m):
        idx = list("1"*(l-1)+"0"*(m-l+1))
        idx[i] = "1"
        logical_indices.append(N-1-bin2int("".join(idx)[::-1]))
    for i in range(N):
        if bin_wt(N-1-i) < l: # |+>
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
    
    logical_indices = []
    for i in range(l-1, m):
        idx = list("1"*(l-1)+"0"*(m-l+1))
        idx[i] = "1"
        logical_indices.append(N-1-bin2int("".join(idx)[::-1]))

    for i in range(N):
        if bin_wt(N-1-i) < l: # |+>
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

stab = []
r = l-1
basis = canonical_min_weight_basis(r, m)
print(f"RM({r},{m}) canonical min-weight basis size = {len(basis)}")
for b in basis:
    stab.append(b['word'])
stab = np.array(stab)

logical_indices_binary = ["000111", "001011", "010011", "100011"]
# logical_indices_binary = ["0011", "0101", "1001"]
logical_indices = [N-1-bin2int(idx) for idx in logical_indices_binary]
print("logical indices", logical_indices)

F = np.array([[1,0],[1,1]])
E = F
for i in range(m-1):
    E = np.kron(E, F)
Lx = E[logical_indices]
Lz = E.T[logical_indices]
# X stabilizer generators: all degree < l monomials
X_stab_indices = [N-1-i for i in range(N) if bin_wt(i) < l]

wt_8_Z_stab_indices_binary = []
for idx in [int2bin(i) for i in range(N) if bin_wt(i) == l]:
    idx = ''.join(map(str, idx))
    if idx not in logical_indices_binary:
        wt_8_Z_stab_indices_binary.append(idx)
print(wt_8_Z_stab_indices_binary)
wt_8_Z_stab_indices = [N-1-bin2int(idx) for idx in wt_8_Z_stab_indices_binary]
wt_8_Hz = E.T[wt_8_Z_stab_indices]


low_wt_Hx = stab[:,::-1] # all of them weight 16
low_wt_Hz = np.vstack((wt_8_Hz, stab))
print("X stabilizer shape", low_wt_Hx.shape)
print("Z stabilizer shape", low_wt_Hz.shape)
assert not (low_wt_Hx @ low_wt_Hz.T % 2).any()