from typing import List, FrozenSet, Dict
from scipy.sparse import csc_matrix
import stim
import numpy as np
import random # random.choice with counts require python>3.11
import time, re, pickle, sys
from collections import  Counter
from functools import reduce
sys.path.append("../")
from PyDecoder_polar import PyDecoder_polar_SCL

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
    return stim.PauliString.from_numpy(xs=xs, zs=zs)

def lists_to_pauli_string(
    x_list: List[int],
    z_list: List[int],
    num_qubits: int,
) -> stim.PauliString:
    xs = np.zeros(num_qubits, dtype=np.bool_)
    zs = np.zeros(num_qubits, dtype=np.bool_)
    for i in x_list: xs[i] = 1
    for i in z_list: zs[i] = 1
    return stim.PauliString.from_numpy(xs=xs, zs=zs)

# helper function: take a PauliString and return its Z-component
def z_component(s):
    x, z = s.to_numpy()
    return stim.PauliString.from_numpy(xs=np.zeros_like(x, dtype=np.bool_), zs=z)
def x_component(s):
    x, z = s.to_numpy()
    return stim.PauliString.from_numpy(xs=x, zs=np.zeros_like(z, dtype=np.bool_))

class AncillaErrorLoader:
    def __init__(self):
        self.N = 64
        with open(f"logs_prep_zero/propagation_dict.pkl", 'rb') as f:
            self.zero_prop_dict = pickle.load(f)
        with open(f"logs_prep_plus/propagation_dict.pkl", 'rb') as f:
            self.plus_prop_dict = pickle.load(f)

    def sample_ancilla_error(self, num_shots, index, parent_dir):
        with open(f"{parent_dir}/{index}_single_fault.pkl", 'rb') as f:
            fault_dict = pickle.load(f)

        with open(f"{parent_dir}/{index}.log", 'r') as f:
            lines = [line for line in f.readlines() if line.startswith("Counter")]
            match = re.search(r'Counter\((\{.*\})\)', lines[0].strip())
            if match:
                counter_dict_str = match.group(1) # extract the dictionary part
                counter_dict = eval(counter_dict_str) # evaluate dict string into dict
                # print(counter_dict)
                counter_obj = Counter(counter_dict)
                num_no_fault = counter_obj[0]
                print("Counter:", counter_obj)
            else:
                sys.exit("Extract counter failed, abort!")

        fault_dict["none"] = num_no_fault

        with open(f"{parent_dir}/{index}_faults.log", 'r') as f:
            lines = f.readlines()
            print(f"{parent_dir}/{index}_faults.log lines length:", len(lines))
            # print(f"number of lines in {index}_faults.log: {len(lines)} (corresponds to order>=2 faults)")
            for line in lines:
                line = line.strip()[1:-1]
                string_values = line.split()
                int_values = tuple(sorted([int(value) for value in string_values]))
                if int_values in fault_dict.keys():  
                    fault_dict[int_values] += 1
                else:
                    fault_dict[int_values] = 1

        print("fault_dict keys distinct entries:", len(fault_dict))
        print("total samples in fault_dict", sum(fault_dict.values()), f"wish to sample {num_shots} samples")

        start = time.time()
        ancilla = random.sample(list(fault_dict.keys()), num_shots, counts=list(fault_dict.values()))
        end = time.time()
        print(f"sampling {num_shots} samples took {end-start} seconds", flush=True)
        return ancilla

    def process_ancilla_error(self, ancilla, state):
        prop_dict = self.plus_prop_dict if state == 'plus' else self.zero_prop_dict

        ancilla_errors = []
        for a in ancilla:
            if a == 'none': # no faults 
                ancilla_errors.append(stim.PauliString(self.N))
            elif isinstance(a, tuple): # multiple fault locations
                ancilla_errors.append(reduce(stim.PauliString.__mul__, [prop_dict[i] for i in a], stim.PauliString(self.N)))
            else: # a single fault
                ancilla_errors.append(prop_dict[a]) # single fault only has residual error one, tested beforehand

        return ancilla_errors   


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

def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False):

    DL_ids: Dict[str, int] = {} # detectors + logical operators
    L_map: Dict[int, FrozenSet[int]] = {} # logical operators
    priors_dict: Dict[int, float] = {} # for each fault

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)
        key = " ".join([f"D{s}" for s in sorted(dets)] + [f"L{s}" for s in sorted(obs)])

        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0

        hid = DL_ids[key]
        L_map[hid] = obs
#         priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            frames: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix({v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")] 
                                       for k, v in DL_ids.items()},
                                      shape=(dem.num_detectors, len(DL_ids)))
    observables_matrix = dict_to_csc_matrix(L_map, shape=(dem.num_observables, len(DL_ids)))
    priors = np.zeros(len(DL_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors