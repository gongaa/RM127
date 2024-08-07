import os
import subprocess
import numpy as np
import sys

r = input('Enter r: ')
n = input('Enter number of samples: ')
l = input('Enter list size: ')
px_min = input('Enter a range of px, px_min: ')
px_max = input('Enter a range of px, px_max (not inclusive): ')
delta_px = 0.001 # increasement
factor = 1000 # for file saving
digits_to_keep = 3
seed = input('Enter random seeds: ')


runtime = input("Enter runtime: ")

path = 'logs'
if not os.path.exists(path):    
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)   

def run_exp(s):
    # dir = 'r'+r+'_l'+l+'_s'+s
    dir = 'r'+r+'_l'+l+'_low_error'
    if not os.path.exists(path+'/'+dir):
        os.mkdir(path+'/'+dir)

    print("Your results will be saved under the directory " + path + "/" + dir)

    for px in np.arange(float(px_min), float(px_max), delta_px):
        cmd = "./build/apps/program -rz "+r+" -l "+l+" -n "+n+" -seed "+s+" -px "+str(round(px,digits_to_keep))
        dest = path + "/" + dir + "/p" + str(int(round(px,digits_to_keep)*factor)) + "_" + s + ".log"
        # dest = path + "/" + dir + "/p" + str(int(round(px,digits_to_keep)*factor)) + ".log"
        # process = subprocess.Popen(['sbatch', '--time', runtime+':00:00', '--wrap', cmd+' &> '+dest])
        process = subprocess.Popen(['sbatch', '--time', runtime+':00:00', '-e', dest, '--wrap', cmd])

for s in seed.split(','):
    run_exp(s)