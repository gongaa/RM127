from typing import List
import stim
import numpy as np
import random # random.choice with counts require python>3.11
import time, re, pickle, sys
from collections import  Counter
from functools import reduce
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
    def __init__(self, decoder_d15, decoder_r2=None, decoder_r4=None):
        self.N = 128
        self.decoder_d15 = decoder_d15 # expect an instance of PyDecoder_polar_SCL(3)
        self.decoder_d7_X = decoder_r2   # expect an instance of PyDecoder_polar_SCL(2)
        self.decoder_d7_Z = decoder_r4   # expect an instance of PyDecoder_polar_SCL(4)
        with open(f"logs_prep_SPAM_equal_CNOT/d15_zero/propagation_dict.pkl", 'rb') as f:
            self.d15_zero_prop_dict = pickle.load(f)
        with open(f"logs_prep_SPAM_equal_CNOT/d15_plus/propagation_dict.pkl", 'rb') as f:
            self.d15_plus_prop_dict = pickle.load(f)
        if self.decoder_d7_X:
            with open(f"logs_prep_SPAM_equal_CNOT/d7_plus/propagation_dict.pkl", 'rb') as f:
                self.d7_plus_prop_dict = pickle.load(f)

    def sample_ancilla_error(self, num_shots, index, parent_dir):
        with open(f"{parent_dir}/{index}_single_fault.pkl", 'rb') as f:
            fault_dict = pickle.load(f)

        with open(f"{parent_dir}/{index}.log", 'r') as f:
            lines = f.readlines()
            target_line = lines[-3].strip()
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

    def process_ancilla_error(self, ancilla, d, state):
        if d==15:
            prop_dict = self.d15_plus_prop_dict if state == 'plus' else self.d15_zero_prop_dict
            decoder_X = self.decoder_d15
            decoder_Z = self.decoder_d15
        elif d==7 and state == 'plus':
            prop_dict = self.d7_plus_prop_dict
            decoder_X = self.decoder_d7_X
            decoder_Z = self.decoder_d7_Z
        else:
            raise NotImplementedError
        ancilla_errors = []
        for a in ancilla:
            if a == 'none': # no faults 
                ancilla_errors.append(stim.PauliString(self.N))
            elif isinstance(a, tuple): # multiple fault locations
                ancilla_errors.append(reduce(stim.PauliString.__mul__, [prop_dict[i] for i in a], stim.PauliString(self.N)))
            else: # a single fault
                ancilla_errors.append(prop_dict[a]) # single fault only has residual error one, tested beforehand (TODO)

        for i in range(len(ancilla_errors)):
            faults = ancilla[i]
            if not isinstance(faults, tuple):
                continue
            residual_error = ancilla_errors[i]
            if residual_error.weight > len(faults):
                x_component = residual_error.pauli_indices('XY')
                z_component = residual_error.pauli_indices('YZ')
                x_num_flip = decoder_X.decode(x_component)
                x_class_bit = decoder_X.last_info_bit
                x_corr = decoder_X.correction
                if state == 'zero' and x_class_bit == 1:
                    print(f"ALERT! X-flip causing logical X errors", flush=True)
                z_num_flip = decoder_Z.decode(z_component)
                z_class_bit = decoder_Z.last_info_bit
                z_corr = decoder_Z.correction
                if state == 'plus' and z_class_bit == 1:
                    print(f"ALERT! Z-flip causing logical Z errors", flush=True)
                total_wt = len(set(x_corr) | set(z_corr))
                print(f"fault {faults}, error before reduction XY: {x_component}, YZ: {z_component}, after reduction wt {total_wt}, XY: {x_corr}, YZ: {z_corr}", flush=True)
                ancilla_errors[i] = lists_to_pauli_string(x_corr, z_corr, self.N)

        return ancilla_errors   