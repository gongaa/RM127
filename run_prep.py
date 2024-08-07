import os
import subprocess
import numpy as np
import sys

n = input('Enter number of experiments: ')
while True:
    d = input('Enter distance (7,15): ')
    if d in ['7', '15']:
        break
while True:
    state = input('Enter state (zero, plus): ')
    if state in ['zero', 'plus']:
        break

runtime = input("Enter runtime: ")
suffix = "d"+d+"_"+state
filename = "full_prep_sim_" + suffix + ".py"
path = "logs_prep_" + suffix
if not os.path.exists(path):    
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)   

def run_exp(s):

    print("Your results will be saved under the directory " + path + "/") 

    cmd = "python " + filename
    dest = path + "/" + str(s) + ".log"
    process = subprocess.Popen(['sbatch', '--mem-per-cpu', '3000', '--time', runtime+':00:00', '--output', dest, '--wrap', cmd])

for s in range(int(n)):
    run_exp(s)