from PyDecoder_polar import PyDecoder_polar_SCL
import os
import glob
import re

num_fault_pattern = re.compile(r"^(\d+) faults occurred")
pattern = re.compile(r"^final wt on output .* X: \[(.*?)\], Y: \[(.*?)\], Z: \[(.*?)\]")
# pattern = re.compile(r"^final wt after copying: .* X: \[(.*?)\], Y: \[(.*?)\], Z: \[(.*?)\]")
d = 15
rz = 4 if d==7 else 3
rx = 2 if d==7 else 3
decoder_X = PyDecoder_polar_SCL(rx)
decoder_Z = PyDecoder_polar_SCL(rz)
state = "zero"
    
# test if strict FT is ever violated in simulations
# the residual error was calculated at the runtime
# if it is higher than the fault-order, reduce it using decoder and compare again
log_files = glob.glob(os.path.join('logs_prep_SPAM_equal_CNOT', f"d{d}_{state}" , "*", '*.log'))
for log_file in log_files:
    # print(log_file)
    with open(log_file, 'r') as file:
        num_fault = None
        for line in file:
            if line.startswith("2 faults occurred"):
                print(f"ALERT! two faults occurred causing high-weight error, log file {log_file}")
            num_fault_match = num_fault_pattern.match(line)
            if num_fault_match:
                num_fault = int(num_fault_match.group(1))
            match = pattern.match(line)
            if match:
                x_list = list(filter(lambda s: s != '', match.group(1).split(',')))
                y_list = list(filter(lambda s: s != '', match.group(2).split(',')))
                z_list = list(filter(lambda s: s != '', match.group(3).split(',')))
                x_list = [int(x.strip()) for x in x_list]
                y_list = [int(y.strip()) for y in y_list]
                z_list = [int(z.strip()) for z in z_list]
                # print(x_list, y_list, z_list)
                new_x_list = x_list + y_list
                if len(new_x_list) > (d-1)//2: # check if wt smaller up to stabilizer
                    num_flip = decoder_X.decode(new_x_list)
                    class_bit = decoder_X.last_info_bit
                    new_x_list = decoder_X.correction
                    # print(f"degeneracy #X flip: {num_flip}, class bit {class_bit} (can be 0/1 if in plus state)")
                    if num_flip > 2:
                        print(f"X-flip weight > 2, #flip={num_flip}, #fault={num_fault}, log file {log_file}")
                        print(x_list, y_list, z_list)
                    if state == 'zero' and class_bit == 1:
                        print(f"ALERT! X-flip causing logical X errors, log file {log_file}")
                new_z_list = z_list + y_list
                if len(new_z_list) > (d-1)//2:
                    num_flip = decoder_Z.decode(new_z_list)
                    class_bit = decoder_Z.last_info_bit
                    new_z_list = decoder_Z.correction
                    # print(f"degeneracy #Z flip: {num_flip}, class bit {class_bit} (can be 0/1 if in zero state)")
                    if num_flip > 2:
                        print(f"Z-flip weight > 2, #flip={num_flip}, #fault={num_fault}, log file {log_file}")
                        print(x_list, y_list, z_list)
                    if state == 'plus' and class_bit == 1:
                        print(f"ALERT! Z-flip causing logical Z errors, log file {log_file}")
                total_wt = len(set(new_x_list) | set(new_z_list))
                if total_wt > 2:
                    print(f"total weight {total_wt}, num fault {num_fault}, log file {log_file}")
                    

'''
# same purpose as above: still testing if strict FT is ever violated in simulations.
# difference is that this one recomputes the residual error using the propagation dict, instead of using the residual error computed during runtime.
# this version takes much longer to run.
log_files = [log_file for log_file in log_files if "faults" in log_file]
import pickle
with open(f"logs_prep_SPAM_equal_CNOT/d{d}_{state}/propagation_dict.pkl", 'rb') as f:
    prop_dict = pickle.load(f)

N = 128
from functools import reduce
import stim
for log_file in log_files:
    print(log_file)
    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()[1:-1]
            string_values = line.split()
            int_values = [int(value) for value in string_values]
            residual_error = reduce(stim.PauliString.__mul__, [prop_dict[i] for i in int_values], stim.PauliString(N))
            if residual_error.weight > len(int_values):
                # print("before degeneracy reduction, residual error weight higher than fault-order in file", log_file, int_values, residual_error.weight)
                x_component = residual_error.pauli_indices('XY')
                z_component = residual_error.pauli_indices('YZ')
                # print(f"X component len: {len(x_component)}, Z component len: {len(z_component)}")
                num_flip = decoder_X.decode(x_component)
                class_bit = decoder_X.last_info_bit
                x_corr = decoder_X.correction
                if state == 'zero' and class_bit == 1:
                    print(f"ALERT! X-flip causing logical X errors")
                num_flip = decoder_Z.decode(z_component)
                class_bit = decoder_Z.last_info_bit
                z_corr = decoder_Z.correction
                if state == 'plus' and class_bit == 1:
                    print(f"ALERT! Z-flip causing logical Z errors")
                # print(f"after degeneracy reduction, X wt {len(x_corr)}, Z wt {len(z_corr)}")
                # print(f"after degeneracy reduction, X corr {x_corr}, Z corr {z_corr}")
                total_wt = len(set(x_corr) | set(z_corr))
                if total_wt > len(int_values):
                    print(f"ALERT: total weight {total_wt}, num fault {len(int_values)}, log file {log_file}, DEM columns {int_values}")
'''         

'''
# counting ratio of order-s faults in all the accepted samples
# Figure 5.4 in my thesis
from collections import Counter
import sys
import numpy as np
cnt_dict = {}
for p in [0.005,0.004,0.003,0.002,0.0015,0.001]:
    log_files = glob.glob(os.path.join('logs_prep_SPAM_equal_CNOT', f"d{d}_{state}" , f"p{str(p).split('.')[1]}", '*.log'))
    log_files = [log_file for log_file in log_files if "faults" not in log_file]
    total_count = Counter()
    for log_file in log_files:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            target_line = lines[-3].strip() 
            match = re.search(r'Counter\((\{.*\})\)', target_line)
            if match:
                counter_dict_str = match.group(1) # extract the dictionary part
                counter_dict = eval(counter_dict_str) # evaluate dict string into dict
                # print(counter_dict)
                total_count += Counter(counter_dict)
            else:
                print(log_file)
                sys.exit("Extract counter failed, abort!")
    cnt_dict[p] = total_count

for p in [0.005,0.004,0.003,0.002,0.0015,0.001]:
    ratio = []
    cnt = cnt_dict[p]
    for i in range(7):
        ratio.append(float(cnt[i])/cnt.total())
    print(f"{p}", " ".join([f"{r:.4e}" for r in ratio]))
'''