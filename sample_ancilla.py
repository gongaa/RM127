import pickle
import re
from collections import Counter
import random
import time
import stim
import numpy as np
from functools import reduce

N = 2 ** 7

def sample_ancilla_error(num_shots, d, state, index, factor=1.0):
    if factor == 1.0:
        with open(f"logs_prep_single_equal_CNOT/d{d}_{state}/propagation_dict.pkl", 'rb') as f:
            prop_dict = pickle.load(f)
    else:
        with open(f"logs_prep_single_half_CNOT/d{d}_{state}/propagation_dict.pkl", 'rb') as f:
            prop_dict = pickle.load(f)

    parent_dir = "logs_prep_single_equal_CNOT" if factor == 1.0 else "logs_prep_single_half_CNOT"
    parent_dir += f"/d{d}_{state}"

    with open(f"{parent_dir}/{index}_single_fault.pkl", 'rb') as f:
        fault_dict = pickle.load(f)

    with open(f"{parent_dir}/{index}.log", 'r') as f:
        lines = f.readlines(0)
        target_line = lines[-2].strip()
        match = re.search(r'Counter\((\{.*\})\)', target_line)
        if match:
            # print("group 0", match.group(0))
            counter_dict_str = match.group(1) # extract the dictionary part
            counter_dict = eval(counter_dict_str) # evaluate dict string into dict
            # print(counter_dict)
            counter_obj = Counter(counter_dict)
            num_no_fault = counter_obj[0]

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

    # print(len(fault_dict))

    start = time.time()
    ancilla = random.sample(list(fault_dict.keys()), num_shots, counts=list(fault_dict.values()))
    # print(ancilla)
    end = time.time()
    print(f"sampling {num_shots} samples elapsed time {end-start} seconds")

    ancilla_errors = []
    for a in ancilla:
        if a == 'none': # no faults 
            ancilla_errors.append(stim.PauliString(N))
        elif isinstance(a, tuple): # multiple fault locations
            err = reduce(stim.PauliString.__mul__, [prop_dict[i] for i in a], stim.PauliString(N))
            print("multiple fault", err)
            ancilla_errors.append(err)
        elif isinstance(a, np.int64): # a single fault
            ancilla_errors.append(prop_dict[a])
            # print("single fault", prop_dict[a])
        else:
            print("key is of unknown types", type(a))

    wts = [e.weight for e in ancilla_errors]
    # print(wts)
    # print(wts.max(), wts.min())
    return ancilla_errors

# sample_ancilla_error(200*1024, 15, 'zero', 0)