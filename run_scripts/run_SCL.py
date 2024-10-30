import os
import subprocess
import numpy as np

r = input('Enter r: ')
n = input('Enter number of samples: ')
l = input('Enter list size: ')
px_min = input('Enter a range of px, px_min: ')
px_max = input('Enter a range of px, px_max (not inclusive): ')
delta_px = 0.001 # increasement
seed = input('Enter random seeds: ')

# r = 4
# n = 10000000
# l = 8


runtime = input("Enter runtime: ")

path = 'logs'
if not os.path.exists(path):    
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)   

def run_exp(s):
    dir = f"r{r}_l{l}_low_error"
    if not os.path.exists(path+'/'+dir):
        os.mkdir(path+'/'+dir)

    print("Your results will be saved under the directory " + path + "/" + dir)

    for px in np.arange(float(px_min), float(px_max), delta_px):
        cmd = f"./build/apps/program -rz {r} -l {l} -n {n} -seed {s} -px {px}"
        dest = f"{path}/{dir}/p{str(px).split('.')[1]}_s{s}.log"
        process = subprocess.Popen(['sbatch', '--time', runtime+':00:00', '-e', dest, '--wrap', cmd])

for s in seed.split(','):
    run_exp(s)

# for s in range(20):
#     run_exp(s)