import stim
print(stim.__version__)
import numpy as np
from typing import List
from pprint import pprint
import time
from scipy.sparse import csc_matrix
import operator
import itertools
import random
from functools import reduce
from PyDecoder_polar import PyDecoder_polar_SCL

n = 5 
N = 2 ** n
# wt_thresh = n - (n-1)//3 # for [[31,1,3]]
wt_thresh = n - (n-1)//2 # for [[31,1,7]]

decoder = PyDecoder_polar_SCL(n-wt_thresh, n)
def is_malignant(s, order):
    num_flip = decoder.decode(list(np.nonzero(s)[0]))
    class_bit = decoder.last_info_bit
    is_malignant = False
    if num_flip > order or (class_bit==1 and state=='+' and flip_type==1) or\
                           (class_bit==1 and state=='0' and flip_type==0):
        is_malignant = True
    # print(f"original wt: {s.sum()}, up to stabilizer: {num_flip}, is malignant: {is_malignant}")
    return is_malignant

bin_wt = lambda i: bin(i)[2:].count('1')
bit_rev = lambda t: int(bin(t)[2:].rjust(n, '0')[::-1], 2)

frozen_mask = [bin_wt(i)<wt_thresh for i in range(N)]
frozen_mask[-1] = True # logical |0>

int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)

def ce(exclude, l=0, u=n): # choose except
    choices = set(range(l,u)) - set([exclude])
    return random.choice(list(choices))

def Eij(i,j):
    A = np.eye(n, dtype=int)
    A[i,j] = 1
    return A

def is_codeword(M):
    unencode = (M @ E) % 2
    return np.logical_not(unencode[:, frozen_mask].any(axis=1))
    
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

def dict_to_csc_matrix(elements_dict, shape):
    # Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict` 
    # giving the indices of nonzero rows in each column.
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)

def dem_to_check_matrices(dem: stim.DetectorErrorModel, circuit, num_detector, tick_circuits, flip_type):
    # set flip_type to 0 for X-flips and 1 for Z-flips
    explained_errors: List[stim.ExplainedError] = circuit.explain_detector_error_model_errors(dem_filter=dem, reduce_to_one_representative_error=False)
    
    D_ids: Dict[str, int] = {} # detectors operators
    priors_dict: Dict[int, float] = {} # for each fault
    error_dict = {} # for where the fault happened
    residual_error_dict = {}

    def handle_error(prob: float, detectors: List[int], rep_loc) -> None:
        dets = frozenset(detectors)
        key = " ".join([f"D{s}" for s in sorted(dets)])

        if key not in D_ids:
            D_ids[key] = len(D_ids)
            priors_dict[D_ids[key]] = 0.0

        hid = D_ids[key]
#         priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        priors_dict[hid] += prob
        # store error representative location
        error_dict[hid] = rep_loc
        # propagate error to the end of the circuit to create an residual fault PCM
        final_pauli_string = propagate(form_pauli_string(rep_loc.flipped_pauli_product), tick_circuits[rep_loc.tick_offset+1:])
        final_wt = final_pauli_string.weight
#         print(rep_loc)
#         print("final pauli string", final_pauli_string, "weight", final_wt)
        residual_error_dict[hid] = final_pauli_string.to_numpy()[flip_type] # for bit flips, use [1] to extract phase flips
        
    index = 0
    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)

#             print(explained_errors[index].circuit_error_locations[0]) ####################### location
            handle_error(p, dets, explained_errors[index].circuit_error_locations[0])
            index += 1
        elif instruction.type == "detector":
#             print("should not have detector, instruction", instruction)
            pass
        elif instruction.type == "logical_observable":
            print("should not have logical observable, instruction", instruction)
            pass
        else:
            raise NotImplementedError()
        
    check_matrix = dict_to_csc_matrix({v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")] 
                                       for k, v in D_ids.items()},
                                      shape=(num_detector, len(D_ids)))
    priors = np.zeros(len(D_ids))
    for i, p in priors_dict.items():
        priors[i] = p

#     print("number of possible residual error strings", len(residual_error_dict))
#     print(np.concatenate([*residual_error_dict.values()]))
    return check_matrix, priors, error_dict, residual_error_dict

def get_pcm(permute, flip_type): # set flip_type to 0 for X-flips, 1 for Z-flips
    p_CNOT = 0.001
    circuit = stim.Circuit()
    tick_circuits = [] # for PauliString.after
    num_detector = 0
    # initialization
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", permute[i])
        else:
            circuit.append("R", permute[i])
    circuit.append("R", N-1)

    for r in range(n): # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [permute[j+i+sep], permute[j+i]])
                    tick_circuit.append("CNOT", [permute[j+i+sep], permute[j+i]])
                    circuit.append("DEPOLARIZE2", [permute[j+i+sep], permute[j+i]], p_CNOT)

        circuit.append("TICK")
        tick_circuits.append(tick_circuit)

    # syndrome detectors
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [j+i+sep, j+i])    
        circuit.append("TICK")

    for i in range(N-1): 
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", i)
        else:
            circuit.append("M", i)
    circuit.append("M", N-1)

    detector_str = ""
    if flip_type == 0: # bit-flips
        for i in range(N): # put detector on the punctured qubit, see if any single fault can trigger it
            if frozen_mask[i]: 
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    else: # phase-flips
        for i in range(N): # put detector on the punctured qubit, see if any single fault can trigger it
            if not frozen_mask[i]: 
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit

    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    pcm, priors, error_explain_dict, residual_error_dict = dem_to_check_matrices(dem, circuit, num_detector, tick_circuits, flip_type)
#     print("flip type", "Z" if flip_type else "X", " #detectors:", num_detector, " residual error shape", len(residual_error_dict))
    pcm = pcm.toarray()
#     if flip_type == 0: # bit-flips
#         print("last detector can be triggered by", pcm[-1,:].sum(), "faults")
    # circuit.diagram('timeline-svg')   
    return pcm, error_explain_dict, residual_error_dict


def get_plus_pcm(permute, flip_type): # set flip_type to 0 for X-flips, 1 for Z-flips
    p_CNOT = 0.001
    circuit = stim.Circuit()
    tick_circuits = [] # for PauliString.after
    num_detector = 0
    # |+> initialization, bit-reversed w.r.t |0>
    for i in range(1,N):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", permute[N-1-i])
        else:
            circuit.append("R", permute[N-1-i])
    circuit.append("RX", N-1-0)

    for r in range(n): # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [permute[j+i], permute[j+i+sep]])
                    tick_circuit.append("CNOT", [permute[j+i], permute[j+i+sep]])
                    circuit.append("DEPOLARIZE2", [permute[j+i], permute[j+i+sep]], p_CNOT)

        circuit.append("TICK")
        tick_circuits.append(tick_circuit)

    # syndrome detectors
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [j+i, j+i+sep])    
        circuit.append("TICK")

    detector_str = ""
    j = 0
    for i in range(1,N)[::-1]: 
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", N-1-i)
            if flip_type == 1: # phase-flips
                detector_str += f"DETECTOR rec[{-N+j}]\n"
                num_detector += 1
        else:
            circuit.append("M", N-1-i)
            if flip_type == 0: # bit-flips
                detector_str += f"DETECTOR rec[{-N+j}]\n"
                num_detector += 1
        j += 1
    circuit.append("MX", N-1-0)
    detector_str += f"DETECTOR rec[-1]\n"; num_detector += 1 # put detector on the punctured qubit

    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit

    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    pcm, priors, error_explain_dict, residual_error_dict = dem_to_check_matrices(dem, circuit, num_detector, tick_circuits, flip_type)
#     print("flip type", "Z" if flip_type else "X", " #detectors:", num_detector, " residual error shape", len(residual_error_dict))
    pcm = pcm.toarray()
#     if flip_type == 1: # phase-flips
#     print("last detector can be triggered by", pcm[-1,:].sum(), "faults")
    # circuit.diagram('timeline-svg')   
    return pcm, error_explain_dict, residual_error_dict 

def test_faults(A, flip_type, state="0"):
    
    a1_permute = [i for i in range(N-1)]
    a1_pcm, a1_error_explain_dict, a1_residual_error_dict = get_pcm(a1_permute, flip_type)
    a1_inv_dict = {}
    a1_num_col = a1_pcm.shape[1]
    for i in range(a1_pcm.shape[1]): 
        key = int(''.join(a1_pcm[:,i].astype('str')), 2)
        if key in a1_inv_dict.keys():
            print("two different faults trigger the same set of detectors")
        else:
            a1_inv_dict[key] = i

    Ax = lambda i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
    a2_permute = [Ax(i) for i in range(N-1)]

    if state == "0":
        a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_pcm(a2_permute, flip_type)
    else:
        a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_plus_pcm(a2_permute, flip_type)


    a2_inv_dict = {}
    a2_num_col = a2_pcm.shape[1]
    for i in range(a2_pcm.shape[1]): 
        key = int(''.join(a2_pcm[:,i].astype('str')), 2)
        if key in a2_inv_dict.keys():
            print("two different faults trigger the same set of detectors")
        else:
            a2_inv_dict[key] = i

#     print("Ancilla 1 inverse dictionary length:", len(a1_inv_dict))
#     print("Ancilla 2 inverse dictionary length:", len(a2_inv_dict))
    
#     print("test one fault on ancilla 1, one fault on ancilla 2")
    for i in range(a1_num_col):
        key = int(''.join(a1_pcm[:,i].astype('str')), 2)
        if key in a2_inv_dict.keys():
            j = a2_inv_dict[key]
            final_error = a1_residual_error_dict[i]
            if final_error.sum() > 1 and is_malignant(final_error, 2):
                return False
                print("final error weight", final_error.sum())
                print("explained faults:")
                print("on ancilla 1,", a1_error_explain_dict[i], "final error at", np.where(a1_residual_error_dict[i])[0])
                print("on ancilla 2,", a2_error_explain_dict[j], "final error at", np.where(a2_residual_error_dict[j])[0])

#     print("test two faults on ancilla 1, one fault on ancilla 2, and create a1 two fault dictionary")
    a1_two_faults_dict = {}
    for i in range(a1_num_col):
        for j in range(i+1, a1_num_col):
            xor = (a1_pcm[:,i] + a1_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a2_inv_dict.keys():
                k = a2_inv_dict[key]
                final_error = a1_residual_error_dict[i] ^ a1_residual_error_dict[j]
                if final_error.sum() > 2 and is_malignant(final_error, 3):
                    return False
                    print("final error weight", final_error.sum())
                    print("explained faults:")
                    print("on ancilla 1,", a1_error_explain_dict[i], "final error at", np.where(a1_residual_error_dict[i])[0])
                    print("on ancilla 1,", a1_error_explain_dict[j], "final error at", np.where(a1_residual_error_dict[j])[0])
                    print("on ancilla 2,", a2_error_explain_dict[k], "final error at", np.where(a2_residual_error_dict[k])[0])
                    
            if key not in a1_two_faults_dict.keys():
                a1_two_faults_dict[key] = a1_residual_error_dict[i] ^ a1_residual_error_dict[j]

#     print("test one fault on ancilla 1, two faults on ancilla 2, and create a2 two fault dictionary")
    a2_two_faults_dict = {}
    for i in range(a2_num_col):
        for j in range(i+1, a2_num_col):
            xor = (a2_pcm[:,i] + a2_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a1_inv_dict.keys():
                k = a1_inv_dict[key]
                final_error = a1_residual_error_dict[k]
                if final_error.sum() > 2 and is_malignant(final_error, 3):
                    return False
                    print("final error weight", final_error.sum())
                    print("explained faults:")
                    print("on ancilla 1,", a1_error_explain_dict[k], "final error at", np.where(a1_residual_error_dict[k])[0])
                    print("on ancilla 2,", a1_error_explain_dict[i], "final error at", np.where(a2_residual_error_dict[i])[0])
                    print("on ancilla 2,", a2_error_explain_dict[j], "final error at", np.where(a2_residual_error_dict[j])[0])
            if key not in a2_two_faults_dict.keys():
                a2_two_faults_dict[key] = a2_residual_error_dict[i] ^ a2_residual_error_dict[j]
    
      
    return True


flip_type = 0
state = '0'
# PA = [(1,0),(2,1),(3,2),(4,3),(0,3)] # lack 4
# list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2
# for _ in range(1000):
#     for a in list(itertools.permutations([1,2,3,4,0])):
#         PB = [(ce(a[0]),a[0]),(ce(a[1]),a[1]),(ce(a[2]),a[2]),(ce(a[3]),a[3]),(ce(a[4]),a[4])]
#         pass_test = test_faults(list_prod(PA) @ list_prod(PB[::-1]) % 2, flip_type)
#         if not pass_test: continue
#         print(PB, ",", flush=True)

PA = [(1,0),(2,1),(3,2),(4,3),(1,3)]
PB = [(2, 1), (1, 4), (2, 0), (4, 3), (4, 2)] 
PC = [(3, 0), (2, 4), (1, 2), (3, 1), (0, 3)]
list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2


# for _ in range(1000):
#     for a in list(itertools.permutations([1,2,3,4])):
#         PC = [(ce(a[0]),a[0]),(ce(a[1]),a[1]),(ce(a[2]),a[2]),(ce(a[3]),a[3])]
#         pass_test = test_faults(list_prod(PA) @ list_prod(PC[::-1]) % 2, flip_type)
#         if not pass_test: continue
#         print(PC, ',')
        
# for _ in range(1000):
#     for a in list(itertools.permutations([1,2,3,4,0])):
#         PC = [(ce(a[0]),a[0]),(ce(a[1]),a[1]),(ce(a[2]),a[2]),(ce(a[3]),a[3]),(ce(a[4]),a[4])]
#         pass_test = test_faults(list_prod(PA) @ list_prod(PC[::-1]) % 2, flip_type)
#         if not pass_test: continue
#         print(PC, ',')
        # print("pass AC")
        # pass_test = test_faults(list_prod(PB) @ list_prod(PC[::-1]) % 2, flip_type, state)
        # if not pass_test: continue
        # print(PC)

for _ in range(1000):
    for a in list(itertools.permutations([0,0,1,3,4])):
        PD = [(ce(a[0]),a[0]),(ce(a[1]),a[1]),(ce(a[2]),a[2]),(ce(a[3]),a[3]),(ce(a[4]),a[4])]
        pass_test = test_faults(list_prod(PA) @ list_prod(PD[::-1]) % 2, flip_type, state)
        if not pass_test: continue
        print("pass AD", flush=True)
        pass_test = test_faults(list_prod(PB) @ list_prod(PD[::-1]) % 2, flip_type, state)
        if not pass_test: continue
        print("pass BD", PD, flush=True)
        pass_test = test_faults(list_prod(PC) @ list_prod(PD[::-1]) % 2, flip_type, state)
        if not pass_test: continue
        print("pass ALL", PD, flush=True)
        
# for _ in range(1000):
#     for a in list(itertools.permutations([1,3,3,4,4])):
#         PA = [(ce(a[0]),a[0]),(ce(a[1]),a[1]),(ce(a[2]),a[2]),(ce(a[3]),a[3]),(ce(a[4]),a[4])]
#         pass_test = test_faults(list_prod(PA) @ list_prod(PB[::-1]) % 2, flip_type, state)
#         if not pass_test: continue
#         pass_test = test_faults(list_prod(PA) @ list_prod(PC[::-1]) % 2, flip_type, state)
#         if not pass_test: continue
#         print(PA,',')