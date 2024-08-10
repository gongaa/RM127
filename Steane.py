import stim
print(stim.__version__)
import numpy as np
import scipy
from scipy.linalg import kron
from typing import List
import time
import operator
from PyDecoder_polar import PyDecoder_polar_SCL


bin_wt = lambda i: bin(i)[2:].count('1')
bit_rev = lambda t: int(bin(t)[2:].rjust(n, '0')[::-1], 2)

n = 7
N = 2 ** n
wt_thresh_d7 = n - (n-1)//3 # for [[127,1,7]]
wt_thresh_d15 = n - (n-1)//2 # for[[127,1,15]]

F = np.array([[1,0],[1,1]])
E = F
for i in range(n-1):
    E = scipy.linalg.kron(E, F)

def propagate(
    pauli_string: stim.PauliString,
    circuits: List[stim.Circuit]
) -> stim.PauliString:
    for circuit in circuits:
        pauli_string = pauli_string.after(circuit)
    return pauli_string

def form_pauli_string(
    flipped_pauli_product: List[stim.GateTargetWithCoords],
    num_qubits: int = N,
) -> stim.PauliString:
    xs = np.zeros(num_qubits, dtype=np.bool_)
    zs = np.zeros(num_qubits, dtype=np.bool_)
    for e in flipped_pauli_product:
        target_qubit, pauli_type = e.gate_target.value, e.gate_target.pauli_type
        if target_qubit >= num_qubits:
            continue
        if pauli_type == 'X':
            xs[target_qubit] = 1
        elif pauli_type == 'Z':
            zs[target_qubit] = 1
        elif pauli_type == 'Y':
            xs[target_qubit] = 1
            zs[target_qubit] = 1
    s = stim.PauliString.from_numpy(xs=xs, zs=zs)
    return s
    
int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)

def get_plus_circuit_d7(circuit, offset):
    for i in range(1,N):
        if bin_wt(i) >= wt_thresh_d7:
            circuit.append("RX", offset+N-1-i)
        else:
            circuit.append("R", offset+N-1-i)
    circuit.append("RX", offset+N-1-0)
    for r in range(n): # rounds
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [offset+j+i, offset+j+i+sep])
    #                 circuit.append("DEPOLARIZE2", [j+i, j+i+sep], p_CNOT)

        circuit.append("TICK")
    return circuit

def get_plus_circuit_d15(circuit, offset):
    for i in range(1,N):
        if bin_wt(i) >= wt_thresh_d15:
            circuit.append("RX", offset+N-1-i)
        else:
            circuit.append("R", offset+N-1-i)
    circuit.append("RX", offset+N-1-0)
    for r in range(n): # rounds
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [offset+j+i, offset+j+i+sep])
    #                 circuit.append("DEPOLARIZE2", [j+i, j+i+sep], p_CNOT)

        circuit.append("TICK")
    return circuit

def get_zero_circuit_d7(circuit, offset):
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh_d7:
            circuit.append("RX", offset+i)
        else:
            circuit.append("R", offset+i)
    circuit.append("R", offset+N-1)

    for r in range(n): # rounds
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [offset+j+i+sep, offset+j+i])
    #                 circuit.append("DEPOLARIZE2", [j+i+sep, j+i], p_CNOT)

        circuit.append("TICK")
    return circuit

def get_zero_circuit_d15(circuit, offset):
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh_d15:
            circuit.append("RX", offset+i)
        else:
            circuit.append("R", offset+i)
    circuit.append("R", offset+N-1)

    for r in range(n): # rounds
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [offset+j+i+sep, offset+j+i])
    #                 circuit.append("DEPOLARIZE2", [j+i+sep, j+i], p_CNOT)

        circuit.append("TICK")
    return circuit

circuit = stim.Circuit()
p_single = 0.05
get_zero_circuit_d15(circuit, offset=0)
get_plus_circuit_d7(circuit, offset=N)
circuit.append("TICK")
for i in range(N-1):
    circuit.append("X_ERROR", i, p_single)
#     circuit.append("X_ERROR", N+i, 0.01)
for i in range(N-1):
    circuit.append("S", i) # use with plus state d=15 on ancilla 1
circuit.append("TICK")
for i in range(N-1):
    circuit.append("CNOT", [i, N+i])
circuit.append("TICK")
for i in range(N-1):
    circuit.append("M", N+i)
circuit.append("TICK")
# unencode to check if in zero state of d=7
unencode_circuit = stim.Circuit()
for i in range(N-1):
    unencode_circuit.append("S_DAG", i)

for r in range(n): # zero state detection
    sep = 2 ** r
    for j in range(0, N, 2*sep):
        for i in range(sep):
            if j+i+sep < N-1:
                unencode_circuit.append("CNOT", [j+i+sep, j+i])
    unencode_circuit.append("TICK")
    
# for r in range(n): # rounds
#     sep = 2 ** r
#     for j in range(0, N, 2*sep):
#         for i in range(sep):
#             if j+i+sep < N-1:
#                 unencode_circuit.append("CNOT", [j+i, j+i+sep])
#     unencode_circuit.append("TICK")

circuit += unencode_circuit
    
frozen_set = np.zeros(N-1, dtype=bool)

for i in range(N-1): # d=7 zero state detection
    if bin_wt(i) >= wt_thresh_d7:
        circuit.append("MX", i)
    else:
        circuit.append("M", i)
        frozen_set[i] = True
        
# for i in range(1,N)[::-1]: # d=7 plus state detection
#     if bin_wt(i) < wt_thresh_d7:
#         circuit.append("M", N-1-i)
#         frozen_set[N-1-i] = True
#     else:
#         circuit.append("MX", N-1-i)

# circuit.diagram('timeline-svg')   

decoder = PyDecoder_polar_SCL(3)
num_shots = 10000
print("number of frozen:", frozen_set.sum())
sampler = circuit.compile_sampler()
result = sampler.sample(shots=num_shots).astype(int)
print(result.shape)
noisy_codeword = result[:,:127]
stab = result[:,127:]
stab = stab[:,frozen_set]
num_error = 0
for i in range(num_shots):
    c = noisy_codeword[i].astype(np.bool_)
    # compare with stab[i]
    num_flip = decoder.decode(list(np.nonzero(c)[0]))
    class_bit = decoder.last_info_bit
    if class_bit == 1:
        c = np.ones(N-1, dtype=np.bool_) ^ c
    c_pauli = stim.PauliString.from_numpy(xs=c.astype(np.bool_), zs=np.zeros(N-1, dtype=np.bool_))
    my_stab = c_pauli.after(unencode_circuit).to_numpy()[0] # only want X component
    if not np.array_equal(my_stab[frozen_set], stab[i]):
        print(f"num_flip: {num_flip}, class_bit: {class_bit}")
        print("incorrect")
        num_error += 1
print(f"#err/#shots: {num_error}/{num_shots}")
