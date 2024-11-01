import os
import subprocess

n = input('Enter number of experiments: ')

runtime = input("Enter runtime: ")

path = 'logs_FT'
if not os.path.exists(path):    
    try:
        os.mkdir(path)
    except OSError as error:
        print(error)   

def run_exp(s):

    print("Your results will be saved under the directory " + path + "/") 

    cmd = "python test_pairs.py"
    dest = path + "/" + str(s) + ".log"
    process = subprocess.Popen(['sbatch', '--time', runtime+':00:00', '--output', dest, '--wrap', cmd])

for s in range(int(n)):
    run_exp(s)