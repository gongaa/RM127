from typing import List
import stim
import numpy as np
import random # random.choice with counts require python>3.11
import time, re, pickle, sys
from collections import  Counter
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

# helper function: take a PauliString and return its Z-component
def z_component(s):
    x, z = s.to_numpy()
    return stim.PauliString.from_numpy(xs=np.zeros_like(x, dtype=np.bool_), zs=z)
def x_component(s):
    x, z = s.to_numpy()
    return stim.PauliString.from_numpy(xs=x, zs=np.zeros_like(z, dtype=np.bool_))

def sample_ancilla_error(num_shots, d, state, index, dir_error_rate, factor=1.0):
    N = 128
    if factor == 1.0:
        with open(f"logs_prep_SPAM_equal_CNOT/d{d}_{state}/propagation_dict.pkl", 'rb') as f:
            prop_dict = pickle.load(f)
    else:
        with open(f"logs_prep_SPAM_half_CNOT/d{d}_{state}/propagation_dict.pkl", 'rb') as f:
            prop_dict = pickle.load(f)

    parent_dir = "logs_prep_SPAM_equal_CNOT" if factor == 1.0 else "logs_prep_SPAM_half_CNOT"
    parent_dir += f"/d{d}_{state}/{dir_error_rate}"

    with open(f"{parent_dir}/{index}_single_fault.pkl", 'rb') as f:
        fault_dict = pickle.load(f)

    with open(f"{parent_dir}/{index}.log", 'r') as f:
        lines = f.readlines(0)
        target_line = lines[-3 if factor==1.0 else -2].strip() # TODO: change to just -3 before release
        match = re.search(r'Counter\((\{.*\})\)', target_line)
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
        lines = f.readlines(0)
        # print(f"number of lines in {index}_faults.log: {len(lines)}")
        for line in lines:
            line = line.strip()[1:-1]
            string_values = line.split()
            int_values = tuple(sorted([int(value) for value in string_values]))
            if int_values in fault_dict.keys():  
                fault_dict[int_values] += 1
            else:
                fault_dict[int_values] = 0

    # print("length of fault_dict to sample from:", len(fault_dict))

    start = time.time()
    ancilla = random.sample(list(fault_dict.keys()), num_shots, counts=list(fault_dict.values()))
    end = time.time()
    # print(f"sampling {num_shots} samples from {parent_dir} took {end-start} seconds")

    ancilla_errors = []
    for a in ancilla:
        if a == 'none': # no faults 
            ancilla_errors.append(stim.PauliString(N))
        elif isinstance(a, tuple): # multiple fault locations
            ancilla_errors.append(reduce(stim.PauliString.__mul__, [prop_dict[i] for i in a], stim.PauliString(N)))
        else: # a single fault
            ancilla_errors.append(prop_dict[a])

    return ancilla_errors

