import stim
print(stim.__version__)
import numpy as np
from itertools import product
import settings
from settings import int2bin, bin2int, bin_wt, append_hypercube_encoding_circuit, append_initialization, append_measurement

m = settings.m
N = settings.N
l = settings.l
K = settings.K
d = settings.d
code = settings.code
Lx = settings.Lx
Lz = settings.Lz
logical_indices = settings.logical_indices

print("logical indices", logical_indices)
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
# Step 1.0: phantom testing, for all elementary matrices (get the permutation correct)
def test_one_block_CNOT_Eij(i, j, initial_state):
    perm = np.eye(m, dtype=int)
    perm[K-1-i, K-1-j] = 1
    perm_Eij = [N-1-bin2int(perm @ np.array(int2bin(N-1-i)) % 2) for i in range(N)]
    circuit = stim.Circuit()
    append_initialization(circuit, initial_state, offset=0, permute=perm_Eij)
    append_hypercube_encoding_circuit(circuit, offset=0, permute=perm_Eij)
    # unencode
    append_hypercube_encoding_circuit(circuit, offset=0)
    # undo the CNOT gate
    circuit.append("CNOT", [logical_indices[i], logical_indices[j]])
    append_measurement(circuit, initial_state, offset=0)
    s = circuit.compile_sampler()
    result = s.sample(shots=100)
    assert not result.any(), "one-block CNOT Eij test: not all zero!"

for i in range(K):
    for j in range(K):
        if j == i: continue
        for initial_state in product(["0","1"], repeat=K):
            test_one_block_CNOT_Eij(i, j, "".join(initial_state))


#####################################################################################
# Step 1.1.0: single cross-block CNOT, between the first logical from both patches
# from patch one to patch two
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
# Step 1.1.1: single cross-block CNOT, between the first logical from both patches, 
# from patch two to patch one
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
print(perm1)

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
# undo the single logical CNOT
circuit.append("CNOT", [N+logical_indices[0], logical_indices[0]])
append_measurement(circuit, "0000", offset=0)
append_measurement(circuit, "++++", offset=N)
diagram = circuit.diagram('timeline-svg')
with open('reversed_CNOT_diagram.svg', 'w') as f:
    print(diagram, file=f)
s = circuit.compile_sampler()
result = s.sample(shots=100)

assert not result.any(), "single CNOT test: not all zero!"
#####################################################################################
# Step 1.1.2: two CNOTs
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
# Step 1.1.3: three CNOTs
# [[0 0 0 0],     [[1 0 0 0],    [[1 0 0 0],
#  [0 1 0 0],  =   [0 1 1 1],  +  [0 0 1 1],
#  [0 0 1 0],      [0 0 1 1],     [0 0 0 1],
#  [0 0 0 1]]      [0 1 0 1]]     [0 1 0 0]]
perm1 = np.array([[1,0,1,0,0,0],
                  [1,1,0,0,0,0],
                  [1,1,1,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]])

perm2 = np.array([[0,0,1,0,0,0],
                  [1,0,0,0,0,0],
                  [1,1,0,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]])

circuit = stim.Circuit()
append_initialization(circuit, "++++", offset=0)
append_initialization(circuit, "0000", offset=N)
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)

for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

# unencode
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
# undo the single logical CNOT
circuit.append("CNOT", [logical_indices[1], N+logical_indices[1]])
circuit.append("CNOT", [logical_indices[2], N+logical_indices[2]])
circuit.append("CNOT", [logical_indices[3], N+logical_indices[3]])
append_measurement(circuit, "++++", offset=0)
append_measurement(circuit, "0000", offset=N)
diagram = circuit.diagram('timeline-svg')
with open('three_CNOT_diagram.svg', 'w') as f:
    print(diagram, file=f)
s = circuit.compile_sampler()
result = s.sample(shots=100)
assert not result.any(), "three CNOT test: not all zero!"





#####################################################################################
# Step 1.2.0: |+000> preparation. SWAP the first logical from |0000> and |++++>
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
# Step 1.2.1: |+000> preparation via three CNOT, measurement, and Pauli correction
perm1 = np.array([[1,0,1,0,0,0],
                  [1,1,0,0,0,0],
                  [1,1,1,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]])

perm2 = np.array([[0,0,1,0,0,0],
                  [1,0,0,0,0,0],
                  [1,1,0,0,0,0],
                  [0,0,0,1,0,0],
                  [0,0,0,0,1,0],
                  [0,0,0,0,0,1]])
circuit = stim.Circuit()
append_initialization(circuit, "++++", offset=0)
append_initialization(circuit, "0000", offset=N)
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)

for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i, i+N])

# unencode patch two
append_hypercube_encoding_circuit(circuit, offset=N)
append_measurement(circuit, "0000", offset=N)

# apply X correction
for log_ind in range(1,4):
    for i in np.where(Lx[log_ind])[0]:
        circuit.append("CX", [stim.target_rec(-4+log_ind), i])

# unencode patch one
append_hypercube_encoding_circuit(circuit, offset=0)
append_measurement(circuit, "+000", offset=0)
diagram = circuit.diagram('timeline-svg')
with open('non_unitary_+000_prep_diagram.svg', 'w') as f:
    print(diagram, file=f)
s = circuit.compile_sampler()
result = s.sample(shots=100)[:,N:]
assert not result.any(), "non unitary +000 prep (one flag) test: not all zero!"
#####################################################################################
# Step 1.2.2: |+000> preparation via one CNOT (from 2 to 1), measurement, and Pauli correction
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

# CNOT 2->1
for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])
for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])

# unencode patch two
append_hypercube_encoding_circuit(circuit, offset=N)
append_measurement(circuit, "++++", offset=N)

# apply Z correction
for log_ind in range(1):
    for i in np.where(Lz[log_ind])[0]:
        circuit.append("CZ", [stim.target_rec(-4+log_ind), i])

# unencode patch one
append_hypercube_encoding_circuit(circuit, offset=0)
append_measurement(circuit, "+000", offset=0)
diagram = circuit.diagram('timeline-svg')
with open('non_unitary_+000_prep_three_flags_diagram.svg', 'w') as f:
    print(diagram, file=f)
s = circuit.compile_sampler()
result = s.sample(shots=100)[:,N:]
assert not result.any(), "non unitary +000 prep (three flags) test: not all zero!"
#####################################################################################
# Step 1.2.2: |000+> preparation via one CNOT (from 2 to 1), measurement, and Pauli correction
circuit = stim.Circuit()
append_initialization(circuit, "0000", offset=0)
append_initialization(circuit, "++++", offset=N)
append_hypercube_encoding_circuit(circuit, offset=0)
append_hypercube_encoding_circuit(circuit, offset=N)
perm1 = np.eye(m, dtype=int)
perm1[1,1] = 0
perm1[0,1] = 1
perm1[1,0] = 1
perm2 = perm1.copy()
perm2[0,0] = 0

# CNOT 2->1
for i in range(N):
    new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])
for i in range(N):
    new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
    new_i = N-1-bin2int(new_i_vec)
    circuit.append("CNOT", [new_i+N, i])

# unencode patch two
append_hypercube_encoding_circuit(circuit, offset=N)
append_measurement(circuit, "++++", offset=N)

# apply Z correction
for log_ind in range(3,4):
    for i in np.where(Lz[log_ind])[0]:
        circuit.append("CZ", [stim.target_rec(-4+log_ind), i])

# unencode patch one
append_hypercube_encoding_circuit(circuit, offset=0)
append_measurement(circuit, "000+", offset=0)
diagram = circuit.diagram('timeline-svg')
with open('non_unitary_000+_prep_three_flags_diagram.svg', 'w') as f:
    print(diagram, file=f)
s = circuit.compile_sampler()
result = s.sample(shots=100)[:,N:]
assert not result.any(), "non unitary 000+ prep (three flags) test: not all zero!"


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
# TODO: Step 2: code switching to the version with fold-S (and back)