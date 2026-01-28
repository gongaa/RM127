import subprocess
import numpy as np

parent_dir = "logs_fold"

p_vec = np.arange(0.01, 0.1, 0.01)[::-1]
num_shots_vec = [1000]*3 + [10000, 100000, 500000, 1000000, 1000000, 1000000]
assert len(p_vec) == len(num_shots_vec), "length not equal"

def run_exp(p_CNOT, num_shots):
    log_file = f"p{str(p_CNOT).split('.')[1]}.log"
    cmd = f"python QRM_fold_simulation.py --num_shots {num_shots} --p {p_CNOT}"
    dest = f"{parent_dir}/{log_file}"
    f = open(dest, "w")
    process = subprocess.Popen([cmd], stdout=f, shell=True)
    # process = subprocess.Popen(['sbatch', '--mem-per-cpu', memory_dict[p_CNOT], '--time', '4:00:00', '--output', dest, '--wrap', cmd])

for (p_CNOT, num_shots) in zip(p_vec, num_shots_vec):
    run_exp(p_CNOT, num_shots)
