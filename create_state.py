import stim
print(stim.__version__)
import numpy as np
import scipy
from scipy.linalg import kron
from typing import List
from pprint import pprint
import time
import operator
from collections import Counter
from functools import reduce



def propagate(
    pauli_string: stim.PauliString,
    circuits: List[stim.Circuit]
) -> stim.PauliString:
    for circuit in circuits:
        pauli_string = pauli_string.after(circuit)
    return pauli_string

def form_pauli_string(
    flipped_pauli_product: List[stim.GateTargetWithCoords],
    num_qubits: int,
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
    

def create_plus(n=7, order=2, offset=0, p_CNOT=0.001, p_meas=0.0005, p_prep=0.0005):

    print(f"Create |+> for n={n}, {'doubly-even' if order==2 else 'triply-even'}, at offset={offset}, p_CNOT={p_CNOT}, p_measure={p_meas}, p_preparation={p_prep}")
    N = 2 ** n
    wt_thresh = n - (n-1)//order
    bin_wt = lambda i: bin(i)[2:].count('1')
    bit_rev = lambda t: int(bin(t)[2:].rjust(n, '0')[::-1], 2)
    int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
    bin2int = lambda l: int(''.join(map(str, l)), 2)
    def Eij(i,j):
        A = np.eye(n, dtype=int)
        A[i,j] = 1
        return A
    # permutations indicated by a list of Eij
    PA = []; PB = []; PC = []; PD = []; PE = []
    if n == 7 and order == 2:
        PA = [(1,2),(6,0),(4,3),(3,6),(0,1),(2,3),(1,6)]
        PB = [(2,6),(5,1),(6,0),(0,5),(4,2),(0,3),(1,4)] 
        PC = [(3,1),(0,2),(2,6),(6,4),(5,0),(6,5),(3,6)] 
        PD = [(5,3),(6,1),(1,2),(2,5),(4,0),(3,4),(4,5)] 
        PE = []
    elif n == 7 and order == 3:
        PA = [(1,0),(2,1),(3,2),(4,3),(5,4)] # lack 5,6
        PB = [(2,6),(5,1),(6,0),(0,5),(4,2)] # lack 3,4
        PE = [(0,3),(3,6),(6,5),(1,4)]
        PC = [(3,1),(0,2),(2,6),(6,4),(5,0)] # lack 3,5
        PD = [(5,3),(6,1),(1,2),(2,5),(4,0)] # lack 4,6

    list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2

    A1 = list_prod(PA[::-1]) % 2
    A2 = list_prod(PB[::-1]) % 2
    A3 = list_prod(PC[::-1]) % 2
    A4 = list_prod(PD[::-1]) % 2
    AE = list_prod(PE) % 2 # important, should be PE, not PE reversed 
    Ax = lambda A, i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
    a1_permute = [Ax(A1, i) for i in range(N-1)]
    a2_permute = [Ax(A2, i) for i in range(N-1)]
    a3_permute = [Ax(A3, i) for i in range(N-1)]
    a4_permute = [Ax(A4, i) for i in range(N-1)]
    PE_permute = [Ax(AE, i) for i in range(N-1)]

    circuit = stim.Circuit()
    error_copy_circuit = stim.Circuit()

    tick_circuits = [] # for PauliString.after

    # ancilla 1
    for i in range(1, N):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset+a1_permute[N-1-i])
            circuit.append("Z_ERROR", offset+a1_permute[N-1-i], p_prep)
        else:
            circuit.append("R", offset+a1_permute[N-1-i])
            circuit.append("X_ERROR", offset+a1_permute[N-1-i], p_prep)
    circuit.append("RX", offset+N-1)

    # ancilla 2
    for i in range(1, N):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset + N + a2_permute[N-1-i])
            circuit.append("Z_ERROR", offset + N + a2_permute[N-1-i], p_prep)
        else:
            circuit.append("R", offset + N + a2_permute[N-1-i])
            circuit.append("X_ERROR", offset + N + a2_permute[N-1-i], p_prep)
    circuit.append("RX", offset+N+N-1)

    # ancilla 3
    for i in range(1, N):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset + 2*N + a3_permute[N-1-i])
            circuit.append("Z_ERROR", offset + 2*N + a3_permute[N-1-i], p_prep)
        else:
            circuit.append("R", offset + 2*N + a3_permute[N-1-i])
            circuit.append("X_ERROR", offset + 2*N + a3_permute[N-1-i], p_prep)
    circuit.append("RX", offset+2*N+N-1)

    # ancilla 4
    for i in range(1, N):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset + 3*N + a4_permute[N-1-i])
            circuit.append("Z_ERROR", offset + 3*N + a4_permute[N-1-i], p_prep)
        else:
            circuit.append("R", offset + 3*N + a4_permute[N-1-i])
            circuit.append("X_ERROR", offset + 3*N + a4_permute[N-1-i], p_prep)
    circuit.append("RX", offset+3*N+N-1)

    circuit.append("TICK")

    for r in range(n): # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [offset+a1_permute[j+i], offset+a1_permute[j+i+sep]])
                    tick_circuit.append("CNOT", [offset+a1_permute[j+i], offset+a1_permute[j+i+sep]])
                    circuit.append("DEPOLARIZE2", [offset+a1_permute[j+i], offset+a1_permute[j+i+sep]], p_CNOT)
                    circuit.append("CNOT", [offset + N + a2_permute[j+i], offset + N + a2_permute[j+i+sep]])
                    tick_circuit.append("CNOT", [offset + N + a2_permute[j+i], offset + N + a2_permute[j+i+sep]])
                    circuit.append("DEPOLARIZE2", [offset + N + a2_permute[j+i], offset + N + a2_permute[j+i+sep]], p_CNOT)
                    circuit.append("CNOT", [offset + 2*N + a3_permute[j+i], offset + 2*N + a3_permute[j+i+sep]])
                    tick_circuit.append("CNOT", [offset + 2*N + a3_permute[j+i], offset + 2*N + a3_permute[j+i+sep]])
                    circuit.append("DEPOLARIZE2", [offset + 2*N + a3_permute[j+i], offset + 2*N + a3_permute[j+i+sep]], p_CNOT)
                    circuit.append("CNOT", [offset + 3*N + a4_permute[j+i], offset + 3*N + a4_permute[j+i+sep]])
                    tick_circuit.append("CNOT", [offset + 3*N + a4_permute[j+i], offset + 3*N + a4_permute[j+i+sep]])
                    circuit.append("DEPOLARIZE2", [offset + 3*N + a4_permute[j+i], offset + 3*N + a4_permute[j+i+sep]], p_CNOT)

        circuit.append("TICK")
        tick_circuits.append(tick_circuit)

    # X error detection first
    # copy X error from ancilla 1 to 2, and 3 to 4
    for i in range(N-1):
        circuit.append("CNOT", [offset+i, offset+N+i])
        circuit.append("DEPOLARIZE2", [offset+i, offset+N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [offset+i, offset+N+i])
        circuit.append("CNOT", [offset+2*N+i, offset+2*N+N+i])
        circuit.append("DEPOLARIZE2", [offset+2*N+i, offset+2*N+N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [offset+2*N+i, offset+2*N+N+i])
    circuit.append("TICK")
    tick_circuits.append(error_copy_circuit)

    # in experiments, here one needs to measure ancilla 2 & 4 bitwise
    # add noise to ancilla 2 & 4 here, even though they are already captured by DEPOLARIZE on CNOTs
    for i in range(N-1):
        circuit.append("X_ERROR", offset+N+i, p_meas)
        circuit.append("X_ERROR", offset+3*N+i, p_meas)

    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [offset+N+j+i, offset+N+j+i+sep])    
                circuit.append("CNOT", [offset+3*N+j+i, offset+3*N+j+i+sep])    

    # ancilla 2 bit flip detection
    num_a2_detector = 0
    detector_str = ""
    j = 0
    for i in range(1, N)[::-1]:
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", offset+N+N-1-i)
        else:
            circuit.append("M", offset+N+N-1-i)
            detector_str += f"DETECTOR rec[{-N+j}]\n"
            num_a2_detector += 1
        j += 1
    circuit.append("MX", offset+N+N-1)

    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a2: {num_a2_detector}")


    # ancilla 4 bit flip detection
    num_a4_detector = 0
    detector_str = ""
    j = 0
    for i in range(1, N)[::-1]:
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", offset+3*N+N-1-i)
        else:
            circuit.append("M", offset+3*N+N-1-i)
            detector_str += f"DETECTOR rec[{-N+j}]\n"
            num_a4_detector += 1
        j += 1
    circuit.append("MX", offset+3*N+N-1)
    
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a4: {num_a4_detector}")

    error_copy_circuit = stim.Circuit()
    # copy Z-error from ancilla 1 to 3
    # CNOT pointing from 3 to 1
    for i in range(N-1):
        circuit.append("CNOT", [offset+2*N+i, offset+PE_permute[i]])
        circuit.append("DEPOLARIZE2", [offset+2*N+i, offset+PE_permute[i]], p_CNOT)
        error_copy_circuit.append("CNOT", [offset+2*N+i, offset+PE_permute[i]])
        
    tick_circuits.append(error_copy_circuit)

    # measure ancilla 3 bitwise in X-basis in experiments
    for i in range(N-1):
        circuit.append("Z_ERROR", offset+2*N+i, p_meas)
    # Stim processing for acceptance
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [offset+2*N+j+i, offset+2*N+j+i+sep])    

    # ancilla 3 phase flip detection
    num_a3_detector = 0
    detector_str = ""
    j = 0
    for i in range(1, N)[::-1]:
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", offset+2*N+N-1-i)
            detector_str += f"DETECTOR rec[{-N+j}]\n"
            num_a3_detector += 1
        else:
            circuit.append("M", offset+2*N+N-1-i)
        j += 1
    circuit.append("MX", offset+2*N+N-1)

    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a3: {num_a3_detector}")
    num_flag_detector = num_a2_detector+num_a3_detector+num_a4_detector
    return circuit, num_flag_detector



def create_zero(n=7, order=2, offset=0, p_CNOT=0.001, p_meas=0.0005, p_prep=0.0005):
    print(f"Create |0> for n={n}, {'doubly-even' if order==2 else 'triply-even'}, at offset={offset}, p_CNOT={p_CNOT}, p_measure={p_meas}, p_preparation={p_prep}")
    N = 2 ** n
    wt_thresh = n - (n-1)//order 
    bin_wt = lambda i: bin(i)[2:].count('1')
    bit_rev = lambda t: int(bin(t)[2:].rjust(n, '0')[::-1], 2)
    int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
    bin2int = lambda l: int(''.join(map(str, l)), 2)
    def Eij(i,j):
        A = np.eye(n, dtype=int)
        A[i,j] = 1
        return A
    # permutations indicated by a list of Eij
    PA = []; PB = []; PC = []; PD = []; PE = []
    if n == 7 and order == 2:
        PA = [(1,2),(6,0),(4,3),(3,6),(0,1),(2,3),(1,6)]
        PB = [(2,6),(5,1),(6,0),(0,5),(4,2),(0,3),(1,4)] 
        PC = [(3,1),(0,2),(2,6),(6,4),(5,0),(6,5),(3,6)] 
        PD = [(5,3),(6,1),(1,2),(2,5),(4,0),(3,4),(4,5)] 
        PE = []
    elif n == 7 and order == 3:
        PA = [(1,0),(2,1),(3,2),(4,3),(5,4)] # lack 5,6
        PB = [(2,6),(5,1),(6,0),(0,5),(4,2)] # lack 3,4
        PE = [(0,3),(3,6),(6,5),(1,4)]
        PC = [(3,1),(0,2),(2,6),(6,4),(5,0)] # lack 3,5
        PD = [(5,3),(6,1),(1,2),(2,5),(4,0)] # lack 4,6

    list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2

    A1 = list_prod(PA[::-1]) % 2
    A2 = list_prod(PB[::-1]) % 2
    A3 = list_prod(PC[::-1]) % 2
    A4 = list_prod(PD[::-1]) % 2
    AE = list_prod(PE) % 2 # important, should be PE, not PE reversed 
    Ax = lambda A, i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
    a1_permute = [Ax(A1, i) for i in range(N-1)]
    a2_permute = [Ax(A2, i) for i in range(N-1)]
    a3_permute = [Ax(A3, i) for i in range(N-1)]
    a4_permute = [Ax(A4, i) for i in range(N-1)]
    PE_permute = [Ax(AE, i) for i in range(N-1)]

    circuit = stim.Circuit()
    error_copy_circuit = stim.Circuit()

    tick_circuits = [] # for PauliString.after

    # ancilla 1
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset+a1_permute[i])
            circuit.append("Z_ERROR", offset+a1_permute[i], p_prep)
        else:
            circuit.append("R", offset+a1_permute[i])
            circuit.append("X_ERROR", offset+a1_permute[i], p_prep)
    circuit.append("R", offset + N-1)

    # ancilla 2
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset + N + a2_permute[i])
            circuit.append("Z_ERROR", offset + N + a2_permute[i], p_prep)
        else:
            circuit.append("R", offset + N + a2_permute[i])
            circuit.append("X_ERROR", offset + N + a2_permute[i], p_prep)
    circuit.append("R", offset+N+N-1)

    # ancilla 3
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset + 2*N + a3_permute[i])
            circuit.append("Z_ERROR", offset + 2*N + a3_permute[i], p_prep)
        else:
            circuit.append("R", offset + 2*N + a3_permute[i])
            circuit.append("X_ERROR", offset + 2*N + a3_permute[i], p_prep)
    circuit.append("R", offset+2*N+N-1)

    # ancilla 4
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", offset + 3*N + a4_permute[i])
            circuit.append("Z_ERROR", offset + 3*N + a4_permute[i], p_prep)
        else:
            circuit.append("R", offset + 3*N + a4_permute[i])
            circuit.append("X_ERROR", offset + 3*N + a4_permute[i], p_prep)
    circuit.append("R", offset+3*N+N-1)

    circuit.append("TICK")

    for r in range(n): # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [offset + a1_permute[j+i+sep], offset + a1_permute[j+i]])
                    tick_circuit.append("CNOT", [offset + a1_permute[j+i+sep], offset + a1_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [offset + a1_permute[j+i+sep], offset + a1_permute[j+i]], p_CNOT)
                    circuit.append("CNOT", [offset + N + a2_permute[j+i+sep], offset + N + a2_permute[j+i]])
                    tick_circuit.append("CNOT", [offset + N + a2_permute[j+i+sep], offset + N + a2_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [offset + N + a2_permute[j+i+sep], offset + N + a2_permute[j+i]], p_CNOT)
                    circuit.append("CNOT", [offset + 2*N + a3_permute[j+i+sep], offset + 2*N + a3_permute[j+i]])
                    tick_circuit.append("CNOT", [offset + 2*N + a3_permute[j+i+sep], offset + 2*N + a3_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [offset + 2*N + a3_permute[j+i+sep], offset + 2*N + a3_permute[j+i]], p_CNOT)
                    circuit.append("CNOT", [offset + 3*N + a4_permute[j+i+sep], offset + 3*N + a4_permute[j+i]])
                    tick_circuit.append("CNOT", [offset + 3*N + a4_permute[j+i+sep], offset + 3*N + a4_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [offset + 3*N + a4_permute[j+i+sep], offset + 3*N + a4_permute[j+i]], p_CNOT)

        circuit.append("TICK")
        tick_circuits.append(tick_circuit)


    for i in range(N-1):
        circuit.append("CNOT", [offset+i, offset+N+i])
        circuit.append("DEPOLARIZE2", [offset+i, offset+N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [offset+i, offset+N+i])
        circuit.append("CNOT", [offset+2*N+i, offset+2*N+N+i])
        circuit.append("DEPOLARIZE2", [offset+2*N+i, offset+2*N+N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [offset+2*N+i, offset+2*N+N+i])
    circuit.append("TICK")
    tick_circuits.append(error_copy_circuit)

    # in experiments, here one needs to measure ancilla 2 & 4 bitwise
    # add noise to ancilla 2 & 4 here, even though they are already captured by DEPOLARIZE on CNOTs
    for i in range(N-1):
        circuit.append("X_ERROR", offset+N+i, p_meas)
        circuit.append("X_ERROR", offset+3*N+i, p_meas)
    # and do classical (noisyless) processing to see if accepted
    # Stim unencode is faster than my own implementation, hence I use Stim here
    # unencode of ancilla 2 & 4 for acceptance
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [offset+N+j+i+sep, offset+N+j+i])    
                circuit.append("CNOT", [offset+3*N+j+i+sep, offset+3*N+j+i])    

    # ancilla 2
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", offset+N+i)
        else:
            circuit.append("M", offset+N+i)
    circuit.append("M", offset+N+N-1)

    # bit flip detection
    num_a2_detector = 0
    detector_str = ""
    for i in range(N):
        if bin_wt(i) < wt_thresh:
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a2_detector += 1
    # detector_str += "DETECTOR rec[-1]\n"        
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a2: {num_a2_detector}")

    # ancilla 4
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", offset+3*N+i)
        else:
            circuit.append("M", offset+3*N+i)
    circuit.append("M", offset+3*N+N-1)

    # bit flip detection
    num_a4_detector = 0
    detector_str = ""
    for i in range(N):
        if bin_wt(i) < wt_thresh:
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a4_detector += 1
    # detector_str += "DETECTOR rec[-1]\n"        
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a4: {num_a4_detector}")

    error_copy_circuit = stim.Circuit()
    # copy Z-error from ancilla 1 to 3
    # CNOT pointing from 3 to 1
    for i in range(N-1):
        circuit.append("CNOT", [offset+2*N+i, offset+PE_permute[i]])
        circuit.append("DEPOLARIZE2", [offset+2*N+i, offset+PE_permute[i]], p_CNOT)
        error_copy_circuit.append("CNOT", [offset+2*N+i, offset+PE_permute[i]])
        
    tick_circuits.append(error_copy_circuit)

    # measure ancilla 3 bitwise in X-basis in experiments
    for i in range(N-1):
        circuit.append("Z_ERROR", offset+2*N+i, p_meas)
    # Stim processing for acceptance
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [offset+2*N+j+i+sep, offset+2*N+j+i])    

    # ancilla 3
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", offset+2*N+i)
        else:
            circuit.append("M", offset+2*N+i)
    circuit.append("M", offset+2*N+N-1)


    # phase flip detection
    num_a3_detector = 0
    detector_str = ""
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a3_detector += 1
    # detector_str += "DETECTOR rec[-1]\n"        
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a3: {num_a3_detector}")
    num_flag_detector = num_a2_detector+num_a3_detector+num_a4_detector
    return circuit, num_flag_detector