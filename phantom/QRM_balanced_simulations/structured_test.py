import stim
print(stim.__version__)
import numpy as np
from typing import List, Dict
import time
import itertools
from scipy.sparse import csc_matrix
from PyDecoder_polar import PyDecoder_polar_SCL
from utils import propagate, form_pauli_string
import re
from pathlib import Path
from itertools import product, permutations
from multiprocessing import Process
from utils_matrix import rank
import settings
from settings import int2bin, bin2int, bin_wt, append_hypercube_encoding_circuit, append_initialization, append_measurement

####################### Settings ######################
m = settings.m
N = settings.N
l = settings.l
K = settings.K
d = settings.d
code = settings.code
extra_X_stab_indices = settings.extra_X_stab_indices

state = '+' # prepare the all-plus or all-zero state
flip_type = 0  # 0 for X-flip, 1 for Z-flip
initial_state = ['0'] * N
for i in range(N):
    if bin_wt(N-1-i) < l or (i in extra_X_stab_indices):
        initial_state[i] = "+"
if state == "+": # prepare the all-plus state
    for i in range(l-1, m):
        idx = list("1"*(l-1)+"0"*(m-l+1))
        idx[i] = "1"
        initial_state[N-1-bin2int("".join(idx)[::-1])] = "+"
print(f"initial state: {initial_state.count('+')} set to + and {initial_state.count('0')} set to 0")
#######################################################
print(f"structured_test.py: m={m}, N={N}, K={K}, d={d}")
################## Permutation helpers ################
def sample_plus_perm():
    while True:
        A = np.random.randint(0, 2, (m,m))
        A[4:5,:4] = 0
        A[:5,5:6] = 0
        if rank(A) < m: continue
        return A

def sample_zero_perm():
    while True:
        A = np.random.randint(0, 2, (m,m))
        A[:5,5:6] = 0
        if rank(A) < m: continue
        return A

def extract_6x6_binary_arrays(text: str):
    # 1) get every [[ ... ]] block
    blocks = re.findall(r"\[\[(.*?)\]\]", text, flags=re.DOTALL)

    arrays = []
    for b in blocks:
        # 2) split into rows and clean
        rows = [r.strip() for r in b.strip().splitlines() if r.strip()]
        if len(rows) != 6:
            continue

        parsed = []
        ok = True
        for r in rows:
            # remove everything except digits and whitespace, then split
            cleaned = re.sub(r"[^\d\s]", " ", r)
            toks = cleaned.split()
            if len(toks) != 6:
                ok = False
                break
            nums = list(map(int, toks))
            if not all(n in (0, 1) for n in nums):
                ok = False
                break
            parsed.append(nums)

        if ok:
            arrays.append(np.array(parsed, dtype=int))

    return arrays

#################### PCM helpers ######################
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

            # print(explained_errors[index].circuit_error_locations[0]) ####################### location
            handle_error(p, dets, explained_errors[index].circuit_error_locations[0])
            index += 1
        elif instruction.type == "detector":
            # print("should not have detector, instruction", instruction)
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

    # print("number of possible residual error strings", len(residual_error_dict))
    # print(np.concatenate([*residual_error_dict.values()]))
    return check_matrix, priors, error_dict, residual_error_dict

def get_pcm(permute, flip_type, reverse=False, verbose=False): # set flip_type to 0 for X-flips, 1 for Z-flips
    # prepare all-zero state
    p_CNOT = 0.001
    p_single = 0.0005
    circuit = stim.Circuit()
    tick_circuits = [] # for PauliString.after
    num_detector = 0
    # initialization
    for i in range(N):
        if initial_state[i] == "+":
            circuit.append("RX", permute[i])
            circuit.append("Z_ERROR", permute[i], p_single)
        else:
            circuit.append("R", permute[i])
            circuit.append("X_ERROR", permute[i], p_single)
    circuit.append("TICK")

    r_range = list(range(m))[::-1] if reverse else list(range(m))
    for r in r_range: # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N:
                    circuit.append("CNOT", [permute[j+i+sep], permute[j+i]])
                    tick_circuit.append("CNOT", [permute[j+i+sep], permute[j+i]])
                    circuit.append("DEPOLARIZE2", [permute[j+i+sep], permute[j+i]], p_CNOT)

        circuit.append("TICK")
        tick_circuits.append(tick_circuit)

    # syndrome detectors
    for r in range(m):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [j+i+sep, j+i])    
        circuit.append("TICK")

    for i in range(N): 
        if initial_state[i] == "+":
            circuit.append("MX", i)
        else:
            circuit.append("M", i)

    detector_str = ""
    if flip_type == 0: # bit-flips
        for i in range(N):
            if initial_state[i] == "0":
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    else: # phase-flips
        for i in range(N):
            if initial_state[i] == "+": 
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit

    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    pcm, priors, error_explain_dict, residual_error_dict = dem_to_check_matrices(dem, circuit, num_detector, tick_circuits, flip_type, verbose=verbose)
    # print("flip type", "Z" if flip_type else "X", " #detectors:", num_detector, " residual error shape", len(residual_error_dict))
    pcm = pcm.toarray()
    # diagram = circuit.diagram('timeline-svg')   
    # with open('zero_diagram.svg', 'w') as f:
    #     print(diagram, file=f)

    return pcm, error_explain_dict, residual_error_dict


def get_plus_pcm(permute, flip_type, reverse=False, verbose=False): # set flip_type to 0 for X-flips, 1 for Z-flips
    # prepare all-plus state
    p_CNOT = 0.001
    p_single = 0.0005
    circuit = stim.Circuit()
    tick_circuits = [] # for PauliString.after
    num_detector = 0
    # |+> initialization, bit-reversed w.r.t |0>
    # TODO: remove all bit-reversal (including CNOT direction)
    for i in range(N):
        if initial_state[i] == '+':
            circuit.append("RX", permute[i])
            circuit.append("Z_ERROR", permute[i], p_single)
        else:
            circuit.append("R", permute[i])
            circuit.append("X_ERROR", permute[i], p_single)
    circuit.append("TICK")

    r_range = list(range(m))[::-1] if reverse else list(range(m))
    for r in r_range: # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N:
                    circuit.append("CNOT", [permute[j+i+sep], permute[j+i]])
                    tick_circuit.append("CNOT", [permute[j+i+sep], permute[j+i]])
                    circuit.append("DEPOLARIZE2", [permute[j+i+sep], permute[j+i]], p_CNOT)

        circuit.append("TICK")
        tick_circuits.append(tick_circuit)

    # syndrome detectors
    for r in range(m):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [j+i+sep, j+i])    
        circuit.append("TICK")
    
    for i in range(N): 
        if initial_state[i] == "+":
            circuit.append("MX", i)
        else:
            circuit.append("M", i)

    detector_str = ""
    if flip_type == 0: # bit-flips
        for i in range(N):
            if initial_state[i] == "0":
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    else: # phase-flips
        for i in range(N):
            if initial_state[i] == "+": 
                detector_str += f"DETECTOR rec[{-N+i}]\n"
                num_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit

    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    pcm, priors, error_explain_dict, residual_error_dict = dem_to_check_matrices(dem, circuit, num_detector, tick_circuits, flip_type, verbose=verbose)
    # print("flip type", "Z" if flip_type else "X", " #detectors:", num_detector, " residual error shape", len(residual_error_dict))
    pcm = pcm.toarray()
    # diagram = circuit.diagram('timeline-svg')   
    # with open('plus_diagram.svg', 'w') as f:
    #     print(diagram, file=f)

    return pcm, error_explain_dict, residual_error_dict

########################## MITM across four patches ########################
#    first test      second test
# |0> ---*---------------X----------
#        |               |
# |0> ---X--- MZ         |
#                        |
# |0> ---*---------------*--- MX
#        |
# |0> ---X--- MZ

# |+> ---X---------------*----------
#        |               |
# |+> ---*--- MX         |
#                        |
# |+> ---X---------------X--- MZ
#        |
# |+> ---*--- MX

second_test = False
if state == '0' and flip_type == 1: # zero state, Z flip
    second_test = True
if state == '+' and flip_type == 0: # plus state, X flip
    second_test = True

residual = [0,1] if second_test else [0,2]


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
            to_store = np.zeros(N, dtype=np.bool_)
            if a in residual:
                to_store ^= a_res[i]
            if b in residual:
                to_store ^= b_res[j]
            if key not in dict_0011.keys():
                dict_0011[key] = [to_store]
                explain_dict[key] = (i,j)
            else:
                dict_0011[key].append(to_store)
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
            to_store = np.zeros(N, dtype=np.bool_)
            if a in residual:
                to_store = a_res[i] ^ a_res[j]
            if key not in dict_0002.keys():
                dict_0002[key] = [to_store]
                explain_dict[key] = (i,j)
            else:
                dict_0002[key].append(to_store)
    return dict_0002, explain_dict

if __name__ == "__main__":
    start = time.time()
    sum_2_tuples = [t for t in product(range(2), repeat=4) if sum(t) == 2]
    sum_3_tuples = [t for t in product(range(3), repeat=4) if sum(t) == 3]
    sum_4_tuples = [t for t in product(range(4), repeat=4) if sum(t) == 4]

    perm_0001 = set(permutations((0,0,0,1)))
    perm_0002 = set(permutations((0,0,0,2)))
    perm_0011 = set(permutations((0,0,1,1)))

    sum_1_options = perm_0001
    sum_2_options = perm_0002 | perm_0011

    def split_tuple(t, op1, op2):
        for option in op1:
            remaining = tuple(a-b for a, b in zip(t, option))
            if remaining in op2:
                return option, remaining
        return None

    sum_2_splits = {}
    sum_3_splits = {}
    sum_4_splits = {}

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
        
    decoder = PyDecoder_polar_SCL(l, m, 0 if state == '0' else 1)

    with open(f"{"plus" if state=='+' else "zero"}_triplet_perm.log", "r") as f:
        text = f.read()
        all_arrays = extract_6x6_binary_arrays(text)

    for i in range(len(all_arrays)//3):
        A1 = np.eye(m, dtype=int)
        A2, A3, A4 = all_arrays[3*i:3*i+3]
        Ax = lambda A, i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
        a1_permute = [Ax(A1, i) for i in range(N)]
        a2_permute = [Ax(A2, i) for i in range(N)]
        a3_permute = [Ax(A3, i) for i in range(N)]
        a4_permute = [Ax(A4, i) for i in range(N)]

        print(f"state: {state}, flip type: {'X' if flip_type == 0 else 'Z'}, second test: {second_test}, residual on: {residual}")

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
        
        print(f"finish constructing dictionaries for order-two faults, elapsed time: {time.time() - start} seconds")
        start = time.time()
            

        def get_residual_error(err):
            if flip_type == 0: # X-flip
                num_flip = decoder.decode_X_flip(list(np.nonzero(err)[0]))
            elif flip_type == 1: # Z-flip
                num_flip = decoder.decode_Z_flip(list(np.nonzero(err)[0]))
            return num_flip

        for k, v in sum_2_splits.items():
            if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 2)):
                continue # separate tests on ancilla (1,2) and ancilla (3,4)
            print(f"test 2 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
            a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
            a_exp, b_exp = all_exp_dicts[v[0]], all_exp_dicts[v[1]]
            for k1 in a_dict.keys():
                if k1 in b_dict.keys():
                    final_error = a_dict[k1] ^ b_dict[k1]
                    if final_error.sum() > 2:
                        residual_error_wt = get_residual_error(final_error)
                        if residual_error_wt <= 2: continue
                        i1 = a_exp[k1]
                        j1 = b_exp[k1]
                        print(f"malignant, at columns {i1} {j1}, final error at {np.where(final_error)[0]}, residual error weight {residual_error_wt}")
                        
        num_wt_4 = 0
        num_wt_8 = 0
        for k, v in sum_3_splits.items():
            if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 3)):
                continue # separate tests on ancilla (1,2) and ancilla (3,4)
            print(f"test 3 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
            a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
            a_exp, b_exp = all_exp_dicts[v[0]], all_exp_dicts[v[1]]
            for k1 in a_dict.keys():
                if k1 in b_dict.keys():
                    v1, v2 = a_dict[k1], b_dict[k1]
                    if not isinstance(v1, list): v1 = [v1]
                    if not isinstance(v2, list): v2 = [v2]
                    for (v1_, v2_) in itertools.product(v1, v2):
                        final_error = v1_ ^ v2_
                        if final_error.sum() > 3:
                            residual_error_wt = get_residual_error(final_error)
                            if residual_error_wt <= 3: continue
                            i1 = a_exp[k1]
                            j1, j2 = b_exp[k1]
                            print(f"malignant, at columns {i1} {j1} {j2}, final error at {np.where(final_error)[0]}, residual error weight {residual_error_wt}")
                            if residual_error_wt == 4: num_wt_4 += 1 # really bad faults
                            if residual_error_wt == 8: num_wt_8 += 1 # really bad faults
                        
        print(f"finish testing <= 3 faults, elapsed time: {time.time() - start} seconds")
        print(f"number of order-three faults leading to weight-four residual error: {num_wt_4}; weight-eight residual error: {num_wt_8}")
        start = time.time()

        num_wt_4 = 0
        for k, v in sum_4_splits.items():
            if second_test == False and (((k[0]+k[1]) != 0) and ((k[0]+k[1]) != 4)):
                continue # separate tests on ancilla (1,2) and ancilla (3,4)
            print(f"test 4 faults distributed as {k}, MITM between {v[0]} and {v[1]}")
            a_dict, b_dict = all_res_dicts[v[0]], all_res_dicts[v[1]]
            a_exp, b_exp = all_exp_dicts[v[0]], all_exp_dicts[v[1]]
            for k1 in a_dict.keys():
                if k1 in b_dict.keys():
                    v1, v2 = a_dict[k1], b_dict[k1]
                    for (v1_, v2_) in itertools.product(v1, v2):
                        final_error = v1_ ^ v2_
                        if final_error.sum() > 4: 
                            residual_error_wt = get_residual_error(final_error)
                            if residual_error_wt <= 4: continue
                            i1, i2 = a_exp[k1]
                            j1, j2 = b_exp[k1]
                            print(f"malignant, at columns {i1} {i2} {j1} {j2}, final error at {np.where(final_error)[0]}, residual error weight {residual_error_wt}")
                            num_wt_4 += 1

        print(f"number of order-four faults leading to weight > four residual error: {num_wt_4}")
        print(f"finish testing 4 faults, elapsed time: {time.time() - start} seconds")

