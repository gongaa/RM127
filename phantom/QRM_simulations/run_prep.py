import os
import subprocess

n = input('Enter number of experiments: ')
while True:
    state = input('Enter state (zero, plus): ')
    if state in ['zero', 'plus']:
        break

runtime = input("Enter runtime: ")

# p = input('CNOT error rate: ')

filename = "full_prep_sim_" + state + ".py"
parent_dir = "logs_prep_" + state


def run_exp(s, p):
    path = parent_dir + "/p" + str(p).split('.')[1]
    if not os.path.exists(path):    
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)   
        print("Your results will be saved under the directory " + path + "/") 

    cmd = "python " + filename + f" {s} {p}"
    dest = path + "/" + str(s) + ".log"
    f = open(dest, "w")
    # print(cmd, dest)
    process = subprocess.Popen([cmd], stdout=f, shell=True)
    # process = subprocess.Popen(['sbatch', '--mem-per-cpu', '6000', '--time', runtime+':00:00', '--output', dest, '--wrap', cmd])


for p in [0.001, 0.002, 0.003]:
    path = parent_dir + "/p" + str(p).split('.')[1]
    offset = 120
    for s in range(offset, offset+int(n)):
        if not os.path.exists(f"{path}/{s}_single_fault.pkl"):
            run_exp(s, p)