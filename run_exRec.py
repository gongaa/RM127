import os
import subprocess
import numpy as np
import sys

factor = 1.0
factor_single = 0.2
factor_correction = 1.0
p_CNOT = 0.001
# bs = 1024
bs = 32
rec_type = "H"
num_batch_dict = {0.0002: 10000, 0.0005: 4000, 0.0008: 2000, 0.001: 2000, 0.0015: 3500, 0.002: 800, 0.003:48, 0.004:92}
memory_dict = {0.0002: '3000', 0.0005: '4000', 0.0008: '4000', 0.001: '2000', 0.0015: '2000', 0.002: '1000', 0.003:'1000', 0.004:'1000'}
rec_type_dict = {"H": "Hadamard", "S": "S", "CNOT": "CNOT", "T": "code_switch"}
parent_dir = "logs_exRec"

def run_exp(factor, factor_single, factor_correction, p_CNOT, rec_type, index):

    d1 = "SPAM_equal_CNOT" if factor == 1.0 else "SPAM_half_CNOT"
    d2 = rec_type_dict[rec_type]
    log_file = f"s{int(10*factor_single)}_c{int(10*factor_correction)}_p{str(p_CNOT).split('.')[1]}_index{index}.log"
    num_batch = num_batch_dict[p_CNOT]
    cmd = f"python full_Steane.py -fs {factor_single} -fc {factor_correction} --num_batch {num_batch} --p_CNOT {p_CNOT} -t {rec_type} --index {index} -bs {bs}"
    dest = f"{parent_dir}/{d1}/{d2}/{log_file}"
    # print(cmd, dest)
    process = subprocess.Popen(['sbatch', '--mem-per-cpu', memory_dict[p_CNOT], '--time', '4:00:00', '--output', dest, '--wrap', cmd])

for factor in [1.0]:
    for factor_correction in [0.0]:
        # for p_CNOT in [0.0005, 0.0008, 0.001, 0.002, 0.003, 0.004]:
        for p_CNOT in [0.004]:
        # for p_CNOT in [0.0015]:
            # for rec_type in ["H", "S", "CNOT", "T"]:
            for rec_type in ["CNOT"]:
                for index in range(125):
                # for index in [205,209]:
                    run_exp(factor, factor_single, factor_correction, p_CNOT, rec_type, index)

# for factor in [0.5]:
#     for factor_correction in [0.2]:
#         for p_CNOT in [0.0005, 0.0008, 0.001, 0.002, 0.003, 0.004]:
#             run_exp(factor, factor_single, factor_correction, p_CNOT, "T")