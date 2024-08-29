import stim
print(stim.__version__)
import numpy as np
from typing import List
import time
from scipy.sparse import csc_matrix
import operator
import itertools
import random
from functools import reduce
from PyDecoder_polar import PyDecoder_polar_SCL
from utils import propagate, form_pauli_string
import pickle
from pathlib import Path
from multiprocessing import Process

n = 7
N = 2 ** n
d = 15
if d == 15:
    wt_thresh = n - (n-1)//3 # for [[127,1,7]]
elif d == 15:
    wt_thresh = n - (n-1)//2 # for [[127,1,15]]
else:
    print("unsupported distance", d)

####################### Settings ######################
state = '0'
flip_type = 0  # 0 for X-flip, 1 for Z-flip
# d = 15
if d == 15:
    if state == '0' and flip_type == 1: # for zero state X-flip test fist (separately on A1A2 and A3A4), then Z-flip test (errors from all four ancilla can contribute)
        second_test = True
    elif state == '+' and flip_type == 0: # for plus state Z-flip test first
        second_test = True
    else:
        second_test = False

if d == 7:
    if flip_type == 1: # for d=7 plus state also bit-flip test first (more detectors)
        second_test = True
    elif flip_type == 0:
        second_test = False

residual = [0,1] if second_test else [0,2]
parent_dir = f"test_strict_FT_state{state}_{'Z' if flip_type else 'X'}"
print("saving dictionaries to directory", parent_dir)
#######################################################

bin_wt = lambda i: bin(i)[2:].count('1')
bit_rev = lambda t: int(bin(t)[2:].rjust(n, '0')[::-1], 2)

int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)

def Eij(i,j):
    A = np.eye(n, dtype=int)
    A[i,j] = 1
    return A

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

def dem_to_check_matrices(dem: stim.DetectorErrorModel, circuit, num_detector, tick_circuits, flip_type, verbose=False):
    # set flip_type to 0 for X-flips and 1 for Z-flips
    explained_errors: List[stim.ExplainedError] = circuit.explain_detector_error_model_errors(dem_filter=dem, reduce_to_one_representative_error=False)
    
    D_ids: Dict[str, int] = {} # detectors operators
    priors_dict: Dict[int, float] = {} # for each fault
    error_dict = {} # for where the fault happened
    residual_error_dict = {}

    def handle_error(prob: float, detectors: List[int], rep_loc) -> None:
        dets = frozenset(detectors)
        if len(dets) == 0:
            print(rep_loc, "triggers no detector")
        key = " ".join([f"D{s}" for s in sorted(dets)])

        if key not in D_ids:
            D_ids[key] = len(D_ids)
            priors_dict[D_ids[key]] = 0.0

        hid = D_ids[key]
        # priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        priors_dict[hid] += prob # ignore second order cancellation
        # store error representative location
        error_dict[hid] = rep_loc
        # propagate error to the end of the circuit to create an residual fault PCM
        final_pauli_string = propagate(form_pauli_string(rep_loc.flipped_pauli_product, N), tick_circuits[rep_loc.tick_offset:])
        final_wt = final_pauli_string.weight
        if verbose:
            print(rep_loc)
            print("final pauli string", final_pauli_string, "weight", final_wt)
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

def get_pcm(permute, flip_type, verbose=False): # set flip_type to 0 for X-flips, 1 for Z-flips
    p_CNOT = 0.001
    p_single = 0.0005
    circuit = stim.Circuit()
    tick_circuits = [] # for PauliString.after
    num_detector = 0
    # initialization
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", permute[i])
            circuit.append("Z_ERROR", permute[i], p_single)
        else:
            circuit.append("R", permute[i])
            circuit.append("X_ERROR", permute[i], p_single)
    circuit.append("R", N-1)
    circuit.append("TICK")

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
            if bin_wt(i) < wt_thresh:
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    else: # phase-flips
        for i in range(N): # put detector on the punctured qubit, see if any single fault can trigger it
            if bin_wt(i) >= wt_thresh: 
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit

    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    pcm, priors, error_explain_dict, residual_error_dict = dem_to_check_matrices(dem, circuit, num_detector, tick_circuits, flip_type, verbose=verbose)
#     print("flip type", "Z" if flip_type else "X", " #detectors:", num_detector, " residual error shape", len(residual_error_dict))
    pcm = pcm.toarray()
#     if flip_type == 0: # bit-flips
#         print("last detector can be triggered by", pcm[-1,:].sum(), "faults")
    # circuit.diagram('timeline-svg')   
    return pcm, error_explain_dict, residual_error_dict


def get_plus_pcm(permute, flip_type, verbose=False): # set flip_type to 0 for X-flips, 1 for Z-flips
    p_CNOT = 0.001
    p_single = 0.0005
    circuit = stim.Circuit()
    tick_circuits = [] # for PauliString.after
    num_detector = 0
    # |+> initialization, bit-reversed w.r.t |0>
    for i in range(1,N):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", permute[N-1-i])
            circuit.append("Z_ERROR", permute[N-1-i], p_single)
        else:
            circuit.append("R", permute[N-1-i])
            circuit.append("X_ERROR", permute[N-1-i], p_single)
    circuit.append("RX", N-1-0)
    circuit.append("TICK")

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
#     detector_str += f"DETECTOR rec[-1]\n"; num_detector += 1 # put detector on the punctured qubit

    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit

    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    pcm, priors, error_explain_dict, residual_error_dict = dem_to_check_matrices(dem, circuit, num_detector, tick_circuits, flip_type, verbose=verbose)
    print("flip type", "Z" if flip_type else "X", " #detectors:", num_detector, " residual error shape", len(residual_error_dict))
    pcm = pcm.toarray()
#     if flip_type == 1: # phase-flips
#     print("last detector can be triggered by", pcm[-1,:].sum(), "faults")
#     circuit.diagram('timeline-svg')   
    return pcm, error_explain_dict, residual_error_dict 

from itertools import product, permutations

sum_2_tuples = [t for t in product(range(2), repeat=4) if sum(t) == 2]
sum_3_tuples = [t for t in product(range(3), repeat=4) if sum(t) == 3]
sum_4_tuples = [t for t in product(range(4), repeat=4) if sum(t) == 4]
sum_5_tuples = [t for t in product(range(5), repeat=4) if sum(t) == 5]
sum_6_tuples = [t for t in product(range(6), repeat=4) if sum(t) == 6]

perm_0001 = set(permutations((0,0,0,1)))
perm_0002 = set(permutations((0,0,0,2)))
perm_0011 = set(permutations((0,0,1,1)))
perm_0003 = set(permutations((0,0,0,3)))
perm_0012 = set(permutations((0,0,1,2)))
perm_0111 = set(permutations((0,1,1,1)))

sum_1_options = perm_0001
sum_2_options = perm_0002 | perm_0011
sum_3_options = perm_0003 | perm_0012 | perm_0111

def split_tuple(t, op1, op2):
    for option in op1:
        remaining = tuple(a-b for a, b in zip(t, option))
        if remaining in op2:
            return option, remaining
    return None

sum_2_splits = {}
sum_3_splits = {}
sum_4_splits = {}
sum_5_splits = {}
sum_6_splits = {}

for t in sum_2_tuples:
    split = split_tuple(t, sum_1_options, sum_1_options)
    if split:
        sum_2_splits[t] = split
        
for k, v in sum_2_splits.items():
    print(k, v)
    
for t in sum_3_tuples:
    split = split_tuple(t, sum_1_options, sum_2_options)
    if split:
        sum_3_splits[t] = split
        
print(f"len of sum_3_splits {len(sum_3_splits)}")
    
for t in sum_4_tuples:
    split = split_tuple(t, sum_2_options, sum_2_options)
    if split:
        sum_4_splits[t] = split
        
print(f"len of sum_4_splits {len(sum_4_splits)}")
    
for t in sum_5_tuples:
    split = split_tuple(t, sum_2_options, sum_3_options)
    if split:
        sum_5_splits[t] = split
        
print(f"len of sum_5_splits {len(sum_5_splits)}")
    
for t in sum_6_tuples:
    split = split_tuple(t, sum_3_options, sum_3_options)
    if split:
        sum_6_splits[t] = split
        
print(f"len of sum_6_splits {len(sum_6_splits)}")

if d == 7:
    PA = [(1,0),(2,1),(3,2),(4,3),(5,4),(0,3),(1,4)]
    PB = [(2,6),(5,1),(6,0),(0,5),(4,2),(0,3),(1,4)] 
    PC = [(3,1),(0,2),(2,6),(6,4),(5,0),(6,5),(3,6)]
    PD = [(5,3),(6,1),(1,2),(2,5),(4,0),(6,5),(3,6)]
elif d == 15:
    PA = [(1,2),(6,0),(4,3),(3,6),(0,1),(2,3),(1,6)]
    PB = [(2,6),(5,1),(6,0),(0,5),(4,2),(0,3),(1,4)] 
    PC = [(3,1),(0,2),(2,6),(6,4),(5,0),(6,5),(3,6)] 
    PD = [(5,3),(6,1),(1,2),(2,5),(4,0),(3,4),(4,5)] 
else:
    PA = []; PB = []; PC = []; PD = []

list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2

A1 = list_prod(PA[::-1]) % 2
A2 = list_prod(PB[::-1]) % 2
A3 = list_prod(PC[::-1]) % 2
A4 = list_prod(PD[::-1]) % 2
Ax = lambda A, i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
a1_permute = [Ax(A1, i) for i in range(N-1)]
a2_permute = [Ax(A2, i) for i in range(N-1)]
a3_permute = [Ax(A3, i) for i in range(N-1)]
a4_permute = [Ax(A4, i) for i in range(N-1)]

if state == '0':
    a1_pcm, a1_error_explain_dict, a1_residual_error_dict = get_pcm(a1_permute, flip_type)
    a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_pcm(a2_permute, flip_type)
    a3_pcm, a3_error_explain_dict, a3_residual_error_dict = get_pcm(a3_permute, flip_type)
    a4_pcm, a4_error_explain_dict, a4_residual_error_dict = get_pcm(a4_permute, flip_type)
else:
    a1_pcm, a1_error_explain_dict, a1_residual_error_dict = get_plus_pcm(a1_permute, flip_type, verbose=False)
    a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_plus_pcm(a2_permute, flip_type)
    a3_pcm, a3_error_explain_dict, a3_residual_error_dict = get_plus_pcm(a3_permute, flip_type)
    a4_pcm, a4_error_explain_dict, a4_residual_error_dict = get_plus_pcm(a4_permute, flip_type)
    
pcms = [a1_pcm, a2_pcm, a3_pcm, a4_pcm]
residual_error_dicts = [a1_residual_error_dict, a2_residual_error_dict, a3_residual_error_dict, a4_residual_error_dict]
explain_dicts = [a1_error_explain_dict, a2_error_explain_dict, a3_error_explain_dict, a4_error_explain_dict]

def construct_0001_dict(a):
    a_pcm = pcms[a]
    a_res = residual_error_dicts[a]
    dict_0001 = {}
    explain_dict = {}
    for i in range(a_pcm.shape[1]):
        key = int(''.join(a_pcm[:,i].astype('str')), 2)
        if key in dict_0001.keys():
            print("two different faults trigger the same set of detectors")
        else:
            to_store = np.zeros(N, dtype=np.bool_)
            if a in residual:
                to_store = a_res[i]
            dict_0001[key] = to_store
            explain_dict[key] = (i)
    return dict_0001, explain_dict

def construct_0011_dict(a, b):
    a_pcm, b_pcm = pcms[a], pcms[b]
    a_res, b_res = residual_error_dicts[a], residual_error_dicts[b]
    dict_0011 = {}
    explain_dict = {}
    for i in range(a_pcm.shape[1]):
        for j in range(b_pcm.shape[1]):
            xor = a_pcm[:,i] ^ b_pcm[:,j]
            key = int(''.join(xor.astype('str')), 2)
            if key not in dict_0011.keys():
                to_store = np.zeros(N, dtype=np.bool_)
                if a in residual:
                    to_store ^= a_res[i]
                if b in residual:
                    to_store ^= b_res[j]
                dict_0011[key] = to_store
                explain_dict[key] = (i,j)
    return dict_0011, explain_dict
                
def construct_0002_dict(a):
    a_pcm = pcms[a]
    a_res = residual_error_dicts[a]
    dict_0002 = {}
    explain_dict = {}
    for i in range(a_pcm.shape[1]):
        for j in range(i+1, a_pcm.shape[1]):
            xor = a_pcm[:,i] ^ a_pcm[:,j]
            key = int(''.join(xor.astype('str')), 2)
            if key not in dict_0002.keys():
                to_store = np.zeros(N, dtype=np.bool_)
                if a in residual:
                    to_store = a_res[i] ^ a_res[j]
                dict_0002[key] = to_store
                explain_dict[key] = (i,j)
    return dict_0002, explain_dict

all_res_dicts = {} # key is stabilizer detector, value is residual on output
all_exp_dicts = {} # to explain the faults
for t in perm_0001:
    print(t)
    [a] = np.where(t)[0]
    print(f"construct dict for one fault on A{a+1}")
    dict_0001, explain_dict = construct_0001_dict(a)
    all_res_dicts[t] = dict_0001
    all_exp_dicts[t] = explain_dict
    
for t in perm_0011:
    if second_test == False and (((t[0]+t[1]) != 0) and ((t[0]+t[1]) != 2)):
        continue # separate tests on ancilla (1,2) and ancilla (3,4)
    print(t)
    a, b = np.where(t)[0]
    print(f"construct dict for one fault on A{a+1}, one fault on A{b+1}")
    dict_0011, explain_dict = construct_0011_dict(a, b)
    all_res_dicts[t] = dict_0011
    all_exp_dicts[t] = explain_dict
    
for t in perm_0002:
    print(t)
    [a] = np.where(t)[0]
    print(f"construct dict for two faults on A{a+1}")
    dict_0002, explain_dict = construct_0002_dict(a)
    all_res_dicts[t] = dict_0002
    all_exp_dicts[t] = explain_dict    
    
decoder = PyDecoder_polar_SCL(3)
def is_malignant(s, order):
    num_flip = decoder.decode(list(np.nonzero(s)[0]))
    class_bit = decoder.last_info_bit
    is_malignant = False
    if num_flip > order or (class_bit==1 and state=='+' and flip_type==1) or\
                           (class_bit==1 and state=='0' and flip_type==0):
        is_malignant = True
    # print(f"original wt: {s.sum()}, up to stabilizer: {num_flip}, is malignant: {is_malignant}")
    return is_malignant

for k, v in sum_2_splits.items():
    if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 2)):
        continue # separate tests on ancilla (1,2) and ancilla (3,4)
    print(f"test 2 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
    a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
    a_exp, b_exp = all_exp_dicts[v[0]], all_exp_dicts[v[1]]
    for k1 in a_dict.keys():
        if k1 in b_dict.keys():
            final_error = a_dict[k1] ^ b_dict[k1]
            if final_error.sum() >= 2 and is_malignant(final_error, 1): # can achieve suppression, i.e., residual weight < fault order
                i1 = a_exp[k1]
                j1 = b_exp[k1]
                print(f"malignant, at columns {i1} {j1}")
                
for k, v in sum_3_splits.items():
    if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 3)):
        continue # separate tests on ancilla (1,2) and ancilla (3,4)
    print(f"test 3 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
    a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
    a_exp, b_exp = all_exp_dicts[v[0]], all_exp_dicts[v[1]]
    for k1 in a_dict.keys():
        if k1 in b_dict.keys():
            final_error = a_dict[k1] ^ b_dict[k1]
            if final_error.sum() >= 3 and is_malignant(final_error, 2): # can achieve suppression
                i1 = a_exp[k1]
                j1, j2 = b_exp[k1]
                print(f"malignant, at columns {i1} {j1} {j2}")
                
for k, v in sum_4_splits.items():
    if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 4)):
        continue # separate tests on ancilla (1,2) and ancilla (3,4)
    print(f"test 4 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
    a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
    a_exp, b_exp = all_exp_dicts[v[0]], all_exp_dicts[v[1]]
    for k1 in a_dict.keys():
        if k1 in b_dict.keys():
            final_error = a_dict[k1] ^ b_dict[k1]
            if final_error.sum() > 4 and is_malignant(final_error, 4):
                i1, i2 = a_exp[k1]
                j1, j2 = b_exp[k1]
                print(f"malignant, at columns {i1} {i2} {j1} {j2}")

def construct_0111_dict(a, b, c, filename):
    my_file = Path(filename)
    if my_file.exists():
        return
    a_pcm, b_pcm, c_pcm = pcms[a], pcms[b], pcms[c]
    a_res, b_res, c_res = residual_error_dicts[a], residual_error_dicts[b], residual_error_dicts[c]
    dict_0111 = {}
#     explain_dict = {}
    for i in range(a_pcm.shape[1]):
        for j in range(b_pcm.shape[1]):
            for k in range(c_pcm.shape[1]):
                xor = a_pcm[:,i] ^ b_pcm[:,j] ^ c_pcm[:,k]
                key = int(''.join(xor.astype('str')), 2)
                if key not in dict_0111.keys():
                    to_store = np.zeros(N, dtype=np.bool_)
                    if a in residual:
                        to_store ^= a_res[i]
                    if b in residual:
                        to_store ^= b_res[j]
                    if c in residual:
                        to_store ^= c_res[k]
                    dict_0111[key] = to_store
#                     explain_dict[key] = (i,j,k)
    with open(filename, 'wb') as f:
        pickle.dump(dict_0111, f)   
   
def construct_0003_dict(a, filename):
    my_file = Path(filename)
    if my_file.exists():
        return
    a_pcm = pcms[a]
    a_res = residual_error_dicts[a]
    dict_0003 = {}
#     explain_dict = {}
    for i in range(a_pcm.shape[1]):
        for j in range(i+1, a_pcm.shape[1]):
            for k in range(j+1, a_pcm.shape[1]):
                xor = a_pcm[:,i] ^ a_pcm[:,j] ^ a_pcm[:,k]
                key = int(''.join(xor.astype('str')), 2)
                if key not in dict_0003.keys():
                    to_store = np.zeros(N, dtype=np.bool_)
                    if a in residual:
                        to_store = a_res[i] ^ a_res[j] ^ a_res[k]
                    dict_0003[key] = to_store
#                     explain_dict[key] = (i,j,k)
    with open(filename, 'wb') as f:
        pickle.dump(dict_0003, f)

def construct_0012_dict(a, b, filename):
    my_file = Path(filename)
    if my_file.exists():
        return
    a_pcm, b_pcm = pcms[a], pcms[b]
    a_res, b_res = residual_error_dicts[a], residual_error_dicts[b]
    dict_0012 = {}
#     explain_dict = {}
    for i in range(a_pcm.shape[1]):
        for j in range(b_pcm.shape[1]):
            for k in range(j+1, b_pcm.shape[1]):
                xor = a_pcm[:,i] ^ b_pcm[:,j] ^ b_pcm[:,k]
                key = int(''.join(xor.astype('str')), 2)
                if key not in dict_0012.keys():
                    to_store = np.zeros(N, dtype=np.bool_)
                    if a in residual:
                        to_store ^= a_res[i]
                    if b in residual:
                        to_store ^= (b_res[j] ^ b_res[k])
                    dict_0012[key] = to_store
#                     explain_dict[key] = (i,j,k)
    with open(filename, 'wb') as f:
        pickle.dump(dict_0012, f)

# check all the stored dict can be correctly loaded
# for a in perm_0012 | perm_0003 | perm_0111:
#     print(a)
#     with open(f"{parent_dir}/{''.join(map(str, a))}.pkl", 'rb') as f:
#         pickle.load(f)

def test_5_faults(t1, t2):
    a_dict = all_res_dicts[t1]
    temp_cnt = 0
    with open(f"{parent_dir}/{''.join(map(str, t2))}.pkl", 'rb') as f:
        b_dict = pickle.load(f)
        for k1 in a_dict.keys():
            if k1 in b_dict.keys():
                final_error = a_dict[k1] ^ b_dict[k1]
                if final_error.sum() > 5 and is_malignant(final_error, 5):
                    temp_cnt += 1
    print(f"found {temp_cnt} sets violating strict FT")

def test_6_faults(t1, t2):
    with open(f"{parent_dir}/{''.join(map(str, t1))}.pkl", 'rb') as f:
        a_dict = pickle.load(f)
    with open(f"{parent_dir}/{''.join(map(str, t2))}.pkl", 'rb') as f:
        b_dict = pickle.load(f)
    temp_cnt = 0
    for k1 in a_dict.keys():
        if k1 in b_dict.keys():
            final_error = a_dict[k1] ^ b_dict[k1]
            if final_error.sum() > 6 and is_malignant(final_error, 6):
                temp_cnt += 1
    print(f"found {temp_cnt} sets violating strict FT")



if __name__ == "__main__":

    for t in perm_0003:
        if second_test == False and (((t[0]+t[1]) != 0) and ((t[0]+t[1]) != 3)):
            continue # separate tests on ancilla (1,2) and ancilla (3,4)
        print(t)
        name = ''.join(map(str, t))
        filename = f"{parent_dir}/{name}.pkl"
        [a] = np.where(t)[0]
        print(f"construct dict for three fault on A{a+1}")
        p = Process(target=construct_0003_dict, args=(a, filename))
        p.start()
        p.join()
        
    for t in perm_0012:
        if second_test == False and (((t[0]+t[1]) != 0) and ((t[0]+t[1]) != 3)):
            continue # separate tests on ancilla (1,2) and ancilla (3,4)
        print(t)
        name = ''.join(map(str, t))
        filename = f"{parent_dir}/{name}.pkl"
        a, b = t.index(1), t.index(2)
        print(f"construct dict for one fault on A{a+1}, two faults on A{b+1}")
        p = Process(target=construct_0012_dict, args=(a, b, filename))
        p.start()
        p.join()

                
    for t in perm_0111:
        if second_test == False and (((t[0]+t[1]) != 0) and ((t[0]+t[1]) != 3)):
            continue # separate tests on ancilla (1,2) and ancilla (3,4)
        print(t)
        name = ''.join(map(str, t))
        filename = f"{parent_dir}/{name}.pkl"
        [a,b,c] = np.where(t)[0]
        print(f"construct dict for one fault on A{a+1}, one fault on A{b+1}, one fault on A{c+1}")
        p = Process(target=construct_0111_dict, args=(a, b, c, filename))
        p.start()
        p.join()


    for k, v in sum_5_splits.items():
        if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 5)):
            continue # separate tests on ancilla (1,2) and ancilla (3,4)
        print(f"test 5 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
        # a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
        p = Process(target=test_5_faults, args=(v[0],v[1]))
        p.start()
        p.join()

    for k, v in sum_6_splits.items():
        if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 6)):
            continue # separate tests on ancilla (1,2) and ancilla (3,4)
        print(f"test 6 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
        # a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
        p = Process(target=test_6_faults, args=(v[0],v[1]))
        p.start()
        p.join()