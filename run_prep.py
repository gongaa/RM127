import os
import subprocess
import numpy as np
import sys

n = input('Enter number of experiments: ')
while True:
    factor = input('Enter ratio p_single / p_CNOT (1.0, 0.5): ')
    if factor in ['1.0', '0.5']:
        break
while True:
    d = input('Enter distance (7,15): ')
    if d in ['7', '15']:
        break
while True:
    state = input('Enter state (zero, plus): ')
    if state in ['zero', 'plus']:
        break
while True:
    phase_first = input('For d=7 plus state, phase first? (y/n): ')
    if phase_first in ['y', 'n']:
        break

runtime = input("Enter runtime: ")

# p = input('CNOT error rate: ')

suffix = "d" + d + "_" + state
if phase_first == 'y' and d == '7':
    suffix += "_phase_first"
filename = "full_prep_sim_" + suffix + ".py"
parent_dir = "logs_prep_SPAM_equal_CNOT/" if factor == '1.0' else "logs_prep_SPAM_half_CNOT/"


def run_exp(s, p):
    path = parent_dir + suffix + "/p" + str(p).split('.')[1]
    if not os.path.exists(path):    
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)   
        print("Your results will be saved under the directory " + path + "/") 

    cmd = "python " + filename + f" {s} {p}"
    dest = path + "/" + str(s) + ".log"
    # print(cmd, dest)
    process = subprocess.Popen(['sbatch', '--mem-per-cpu', '3000', '--time', runtime+':00:00', '--output', dest, '--wrap', cmd])

# for p in [0.005, 0.004, 0.003, 0.002, 0.001, 0.0008, 0.0005, 0.0002, 0.0001]:
# for p in [0.005, 0.004, 0.003, 0.002, 0.001]:
# for p in [0.001,0.002,0.003]:
for p in [0.0015]:
# for p in [0.0005, 0.0008]:
    for s in range(int(n)):
        run_exp(800+s, p)