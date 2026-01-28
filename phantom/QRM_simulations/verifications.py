import stim
print(stim.__version__)
import numpy as np
from itertools import product
import settings
from settings import int2bin, bin2int, bin_wt, append_hypercube_encoding_circuit, append_initialization, append_measurement

# The final goal is to verify the correctness of my single Hadamard scheme
#      |- ----- S -----*---------------*------------ MX ==**
# phi -|  -------------|---------------|------------ MX   ||
#      |- -------------|---------------|------------ MX   ||
#                      |               |                  ||
#      |- ----- S -----X----- S^† -----X----------------| X | -|
# +++ -|  --------------------------------------------------   |- HII|phi>
#      |- --------------------------------------------------  -|

# To verify this, I break down into several steps.
# 1. single cross-block CNOT
# 2. correctness of fold-S: implementing SSII
# 3. correctness of four fold-S: implementing SIII
# 4. Hadamard teleporation

m = settings.m
N = settings.N
l = settings.l
K = settings.K
d = settings.d
logical_indices = []
for i in range(l-1, m):
    idx = list("1"*(l-1)+"0"*(m-l+1))
    idx[i] = "1"
    logical_indices.append(N-1-bin2int("".join(idx)[::-1]))
print(f"Logical indices {logical_indices}, binary representations")
print([int2bin(N-1-i) for i in logical_indices])
# for [[64,4,8]], the four logicals are x3x2x1, x4x2x1, x5x2x1, x6x2x1
F = np.array([[1,0],[1,1]])
E = F
for i in range(m-1):
    E = np.kron(E, F)
Lx = E[logical_indices]
print("logical X generators")
print(Lx)
Lz = E.T[logical_indices]
def Eij(i,j):
    mat = np.eye(m, dtype=int)
    mat[i,j] = 1
    return mat
    
# for fold-S leading to SS
permutation = [i for i in range(N)]
for i in range(N):
    i_bin = int2bin(N-1-i)
    i_bin = [1-i_bin[5], 1-i_bin[4], 1-i_bin[3], 1-i_bin[2], 1-i_bin[1], 1-i_bin[0]]
    permutation[i] = N-1-bin2int(i_bin)
fixed_point = [i for (idx, i) in enumerate(permutation) if i==idx]
print("fixed points", fixed_point)
# need to check INVOLUTION !!!!!!!
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

# for fold-S leading to CZ
permutation = [i for i in range(N)]
for i in range(N):
    i_bin = int2bin(N-1-i)
    i_bin = [1-i_bin[5], 1-i_bin[4], 1-i_bin[2], 1-i_bin[3], 1-i_bin[1], 1-i_bin[0]]
    permutation[i] = N-1-bin2int(i_bin)
fixed_point = [i for (idx, i) in enumerate(permutation) if i==idx]
print("fixed points", fixed_point)
# need to check INVOLUTION !!!!!!!
involution_CZ = []
for i in range(N):
    j = permutation[i]
    if i != j:
        assert permutation[j] == i, "Not an involution!"
        temp = sorted([i,j])
        if temp not in involution_CZ:
            involution_CZ.append(temp)
    else:
        involution_CZ.append([i])

#####################################################################################
# Step 0: test logical X
def test_logical_X(initial_state):
    circuit = stim.Circuit()
    append_initialization(circuit, initial_state, offset=0)
    append_hypercube_encoding_circuit(circuit, offset=0)
    for i in np.where(Lx[0])[0]:
        circuit.append("X", i)
    append_hypercube_encoding_circuit(circuit, offset=0)
    circuit.append("X", logical_indices[0])
    append_measurement(circuit, initial_state, offset = 0)
    s = circuit.compile_sampler()
    result = s.sample(shots=100)
    assert not result.any(), "logical X test: not all zero!"

for initial_state in product(["0","1"], repeat=K):
    test_logical_X("".join(initial_state))
#####################################################################################
# Step 1: single cross-block CNOT, between the first logical from both patches
circuit = stim.Circuit()
append_initialization(circuit, "++++", offset=0)
append_initialization(circuit, "0000", offset=N)
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
perm1 = np.eye(m, dtype=int)
perm1[2,2] = 0
perm1[2,3] = 1
perm1[3,2] = 1
print(perm1)
for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    # print(f"{int2bin(N-1-i)} mapped to {new_i_vec}")
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

perm2 = perm1.copy()
perm2[3,3] = 0
print(perm2)
for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    # print(f"{int2bin(N-1-i)} mapped to {new_i_vec}")
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

# unencode
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
# undo the single logical CNOT
circuit.append("CNOT", [logical_indices[0], N+logical_indices[0]])
append_measurement(circuit, "++++", offset=0)
append_measurement(circuit, "0000", offset=N)
diagram = circuit.diagram('timeline-svg')
with open('CNOT_diagram.svg', 'w') as f:
    print(diagram, file=f)
s = circuit.compile_sampler()
result = s.sample(shots=100)
# print(result)
# print(result.astype(int).sum())
# print(np.where(result))
assert not result.any(), "single CNOT test: not all zero!"

#####################################################################################
# Step 1.1: two CNOTs
# [[1 0 0 0],     [[1 1 0 0],    [[0 1 0 0],
#  [0 1 0 0],  =   [1 0 0 0],  +  [1 1 0 0],
#  [0 0 0 0],      [0 0 1 0],     [0 0 1 0],
#  [0 0 0 0]]      [0 0 0 1]]     [0 0 0 1]]
circuit = stim.Circuit()
append_initialization(circuit, "++++", offset=0)
append_initialization(circuit, "0000", offset=N)
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
perm1 = np.eye(m, dtype=int)
perm1[2,2] = 0
perm1[2,3] = 1
perm1[3,2] = 1
print(perm1)
for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

perm2 = perm1.copy()
perm2[3,3] = 0
perm2[2,2] = 1
print(perm2)
for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

# unencode
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
# undo the single logical CNOT
circuit.append("CNOT", [logical_indices[0], N+logical_indices[0]])
circuit.append("CNOT", [logical_indices[1], N+logical_indices[1]])
append_measurement(circuit, "++++", offset=0)
append_measurement(circuit, "0000", offset=N)
diagram = circuit.diagram('timeline-svg')
with open('CNOT_diagram.svg', 'w') as f:
    print(diagram, file=f)
s = circuit.compile_sampler()
result = s.sample(shots=100)
assert not result.any(), "two CNOT test: not all zero!"

#####################################################################################
# Step 1.2: |+000> preparation. SWAP the first logical from |0000> and |++++>
circuit = stim.Circuit()
append_initialization(circuit, "0000", offset=0)
append_initialization(circuit, "++++", offset=N)
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
perm1 = np.eye(m, dtype=int)
perm1[2,2] = 0
perm1[2,3] = 1
perm1[3,2] = 1
perm2 = perm1.copy()
perm2[3,3] = 0
# CNOT 1->2
# for i in range(N):
#     new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
#     new_i = N-1-bin2int(new_i_vec)
#     circuit.append("CNOT", [new_i, i+N])
# for i in range(N):
#     new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
#     new_i = N-1-bin2int(new_i_vec)
#     circuit.append("CNOT", [new_i, i+N])

# CNOT 2->1
for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])
for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])

# CNOT 1->2
for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])
for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

# CNOT 2->1
for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])
for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])

# unencode
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
append_measurement(circuit, "+000", offset=0)
append_measurement(circuit, "0+++", offset=N)
s = circuit.compile_sampler()
result = s.sample(shots=100)
assert not result.any(), "+000 test: not all zero!"

#####################################################################################
# Step 1.3: one-block CNOT ladder (get the permutation correct)
layer_1 = np.eye(K, dtype=int)
layer_1[3,2] = 1
layer_2 = np.eye(K, dtype=int)
layer_2[3,1] = 1
layer_2[2,0] = 1
perm = np.eye(m, dtype=int)
perm[:K,:K] = layer_2 @ layer_1 % 2
print("CNOT ladder permutation:")
print(perm)
perm_CNOT_ladder = [N-1-bin2int(perm @ np.array(int2bin(N-1-i)) % 2) for i in range(N)]
def test_one_block_CNOT_ladder(initial_state):
    print("CNOT ladder test: initial state", initial_state)
    circuit = stim.Circuit()
    append_initialization(circuit, initial_state, offset=0, permute=perm_CNOT_ladder)
    append_hypercube_encoding_circuit(circuit, offset=0, permute=perm_CNOT_ladder)
    # unencode
    append_hypercube_encoding_circuit(circuit, offset=0)
    # undo the CNOT ladder
    circuit.append("CNOT", [logical_indices[0], logical_indices[2]])
    circuit.append("CNOT", [logical_indices[1], logical_indices[3]])
    circuit.append("CNOT", [logical_indices[0], logical_indices[1]])
    append_measurement(circuit, initial_state, offset=0)
    s = circuit.compile_sampler()
    result = s.sample(shots=100)
    assert not result.any(), "one-block CNOT ladder test: not all zero!"

for initial_state in product(["0","1"], repeat=K):
    test_one_block_CNOT_ladder("".join(initial_state))
#####################################################################################
# Step 2: fold-S for SSII
def test_fold_S(initial_state):
    circuit = stim.Circuit()
    append_initialization(circuit, "++++", offset=0)
    append_hypercube_encoding_circuit(circuit, offset=0)
    for tup in involution:
        if len(tup) == 1:
            # print(f"fixed point at i={tup[0]} with bin {int2bin(N-1-tup[0])}")
            circuit.append("S", tup[0])
        else:
            # print(f"involution pair at i={tup[0]}, j={tup[j]} with bin {int2bin(N-1-tup[0])} and {int2bin(N-1-tup[1])}")
            circuit.append("CZ", [tup[0], tup[1]])
    # unencode
    append_hypercube_encoding_circuit(circuit, offset=0)
    # undo the logical SSII using S dagger's
    circuit.append("S_DAG", logical_indices[0])
    circuit.append("S_DAG", logical_indices[1])
    append_measurement(circuit, "++++", offset=0)

    diagram = circuit.diagram('timeline-svg')
    with open('fold_S_diagram.svg', 'w') as f:
        print(diagram, file=f)
    s = circuit.compile_sampler()
    result = s.sample(shots=100)
    assert not result.any(), "fold-S for SSII test: not all zero!"

for initial_state in product(["0","1"], repeat=K):
    test_fold_S("".join(initial_state))


#####################################################################################
# Step 2: fold-S for CZ
def test_fold_CZ(initial_state):
    circuit = stim.Circuit()
    append_initialization(circuit, initial_state, offset=0)
    append_hypercube_encoding_circuit(circuit, offset=0)
    for tup in involution_CZ:
        if len(tup) == 1:
            # print(f"fixed point at i={tup[0]} with bin {int2bin(N-1-tup[0])}")
            circuit.append("S", tup[0])
        else:
            # print(f"involution pair at i={tup[0]}, j={tup[j]} with bin {int2bin(N-1-tup[0])} and {int2bin(N-1-tup[1])}")
            circuit.append("CZ", [tup[0], tup[1]])
    # unencode
    append_hypercube_encoding_circuit(circuit, offset=0)
    # undo the logical SSII using S dagger's
    circuit.append("CZ", [logical_indices[0], logical_indices[1]])
    append_measurement(circuit, initial_state, offset=0, detector=True)

    diagram = circuit.diagram('timeline-svg')
    with open('fold_S_diagram.svg', 'w') as f:
        print(diagram, file=f)
    s = circuit.compile_sampler()
    result = s.sample(shots=100)
    assert not result.any(), "fold-S for SSII test: not all zero!"

for initial_state in product(["0","1"], repeat=K):
    test_fold_CZ("".join(initial_state))
#
#####################################################################################
# Step 3: four fold-S leading to SIII
# skip this, let us just do HHII using teleportation
def test_HHII(initial_state):
    print("HHII test: initial state", initial_state)
    circuit = stim.Circuit()
    append_initialization(circuit, initial_state, offset=0)
    append_initialization(circuit, "++00", offset=N) # do not vary
    append_hypercube_encoding_circuit(circuit, offset=0)
    append_hypercube_encoding_circuit(circuit, offset=N)
    # fold-S for SSII
    for tup in involution:
        if len(tup) == 1:
            circuit.append("S", tup[0])
            circuit.append("S", N + tup[0])
        else:
            circuit.append("CZ", [tup[0], tup[1]])
            circuit.append("CZ", [N + tup[0], N + tup[1]])

    # two CNOTs
    # for i in range(N):
    #     circuit.append("CNOT", [i, i+N])
    perm1 = np.eye(m, dtype=int)
    perm1[2,2] = 0
    perm1[2,3] = 1
    perm1[3,2] = 1
    for i in range(N):
        new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
        new_i = N-1-bin2int(new_i_vec)
        circuit.append("CNOT", [new_i, i+N])

    perm2 = perm1.copy()
    perm2[3,3] = 0
    perm2[2,2] = 1
    for i in range(N):
        new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
        new_i = N-1-bin2int(new_i_vec)
        circuit.append("CNOT", [new_i, i+N])

    # S dagger on the ++++ block
    for tup in involution:
        if len(tup) == 1:
            circuit.append("S_DAG", N + tup[0])
        else:
            circuit.append("CZ", [N + tup[0], N + tup[1]])

    # four CNOTs
    for i in range(N):
        circuit.append("CNOT", [i, i+N])

    # unencode
    append_hypercube_encoding_circuit(circuit, offset=0)
    append_measurement(circuit, "++++", offset=0)
    # apply X correction
    for log_ind in range(2): # or 2
        for i in np.where(Lx[log_ind])[0]:
            circuit.append("CX", [stim.target_rec(-4+log_ind), N+i]) # or N-1-i
    for log_ind in range(2,4): # or 2
        for i in np.where(Lz[log_ind])[0]:
            circuit.append("CZ", [stim.target_rec(-4+log_ind), N+i]) # or N-1-i


    append_hypercube_encoding_circuit(circuit, offset=N)
    # undo HHII
    circuit.append("H", N + logical_indices[0])
    circuit.append("H", N + logical_indices[1])

    append_measurement(circuit, initial_state, offset=N)
    diagram = circuit.diagram('timeline-svg')
    with open('HHII_diagram.svg', 'w') as f:
        print(diagram, file=f)
    s = circuit.compile_sampler()
    result = s.sample(shots=100)[:,N:]
    # print(result)
    # print(result.astype(int).sum())
    # print(np.where(result))
    assert not result.any(), "HHII test: not all zero!"

for initial_state in product(["0","1"], repeat=K):
    test_HHII("".join(initial_state))
