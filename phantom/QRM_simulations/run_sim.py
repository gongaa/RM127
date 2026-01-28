import subprocess

num_batch_dict = {0.001: 2000, 0.002: 500, 0.003:100}
memory_dict = {0.0002: '3000', 0.0005: '2000', 0.0008: '2000', 0.001: '2000', 0.0015: '2000', 0.002: '1000', 0.003: '1000', 0.004: '1000'}
parent_dir = "logs_sim"

def run_exp(p_CNOT, num_block, index):
    bs = 1024
    dir = f"num_block_{num_block}"
    log_file = f"p{str(p_CNOT).split('.')[1]}_index{index}.log"
    num_batch = num_batch_dict[p_CNOT]
    cmd = f"python CNOT_ladder.py --num_batch {num_batch} --p_CNOT {p_CNOT} -nb {num_block} --index {index} -bs {bs}"
    dest = f"{parent_dir}/{dir}/{log_file}"
    f = open(dest, "w")
    process = subprocess.Popen([cmd], stdout=f, shell=True)
    # process = subprocess.Popen(['sbatch', '--mem-per-cpu', memory_dict[p_CNOT], '--time', '4:00:00', '--output', dest, '--wrap', cmd])

for p_CNOT in [0.001, 0.002, 0.003]:
    for index in range(8):
        run_exp(p_CNOT, num_block=1, index=8+index)
