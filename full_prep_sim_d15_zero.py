import stim
print(stim.__version__)
import numpy as np
from typing import List
import time
import operator
from collections import Counter
from functools import reduce
import sys
import pickle
from utils import propagate, form_pauli_string

n = 7
N = 2 ** n
wt_thresh = n - (n-1)//2 # for [[127,1,15]]

bin_wt = lambda i: bin(i)[2:].count('1')
bit_rev = lambda t: int(bin(t)[2:].rjust(n, '0')[::-1], 2)
int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)

def Eij(i,j):
    A = np.eye(n, dtype=int)
    A[i,j] = 1
    return A
# permutations indicated by a list of Eij
PA = [(1,2),(6,0),(4,3),(3,6),(0,1),(2,3),(1,6)]
PB = [(2,6),(5,1),(6,0),(0,5),(4,2),(0,3),(1,4)] 
PC = [(3,1),(0,2),(2,6),(6,4),(5,0),(6,5),(3,6)] 
PD = [(5,3),(6,1),(1,2),(2,5),(4,0),(3,4),(4,5)] 

list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2

A1 = list_prod(PA[::-1]) % 2
A2 = list_prod(PB[::-1]) % 2
A3 = list_prod(PC[::-1]) % 2
A4 = list_prod(PD[::-1]) % 2

Ax = lambda A, i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
a1_permute = [Ax(A1, i) for i in range(N-1)]
a2_permute = [Ax(A2, i) for i in range(N-1)]
a3_permute = [Ax(A3, i) for i in range(N-1)]
a4_permute = [Ax(A4, i) for i in range(N-1)]

if __name__ == "__main__":
    # Check if an argument has been provided
    if len(sys.argv) != 4:
        print("Usage: python script.py index<int> p_CNOT<float> (p_SPAM/p_CNOT)<float>")
        sys.exit(1)
    try:
        # Get the integer from the command line argument
        input_value = int(sys.argv[1])
        error_rate = float(sys.argv[2])
        factor = float(sys.argv[3]) # 1.0 or 0.5
    except ValueError:
        print("The argument must be an integer.")
        sys.exit(1)
        
    p_CNOT = error_rate
    p_meas = factor * p_CNOT
    p_prep = p_meas
    num_rounds = 2500
    num_shots = 100000
    parent_dir = "logs_prep_SPAM_equal_CNOT/" if factor == 1.0 else "logs_prep_SPAM_half_CNOT/"
    parent_dir += "d15_zero"
    parent_dir += '/p' + str(p_CNOT).split('.')[1] # comment this line out if generating propagation_dict
    print(f"full_prep_sim_d15_zero.py writing to {parent_dir}/{input_value}.log", flush=True)
    print(f"p_CNOT={p_CNOT}, p_measure={p_meas}, p_preparation={p_prep}")

    circuit = stim.Circuit()
    error_copy_circuit = stim.Circuit()

    tick_circuits = [] # for PauliString.after


    # ancilla 1
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", a1_permute[i])
            circuit.append("Z_ERROR", a1_permute[i], p_prep)
        else:
            circuit.append("R", a1_permute[i])
            circuit.append("X_ERROR", a1_permute[i], p_prep)
    circuit.append("R", N-1)

    # ancilla 2
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", N + a2_permute[i])
            circuit.append("Z_ERROR", N + a2_permute[i], p_prep)
        else:
            circuit.append("R", N + a2_permute[i])
            circuit.append("X_ERROR", N + a2_permute[i], p_prep)
    circuit.append("R", N+N-1)

    # ancilla 3
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", 2*N + a3_permute[i])
            circuit.append("Z_ERROR", 2*N + a3_permute[i], p_prep)
        else:
            circuit.append("R", 2*N + a3_permute[i])
            circuit.append("X_ERROR", 2*N + a3_permute[i], p_prep)
    circuit.append("R", 2*N+N-1)

    # ancilla 4
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("RX", 3*N + a4_permute[i])
            circuit.append("Z_ERROR", 3*N + a4_permute[i], p_prep)
        else:
            circuit.append("R", 3*N + a4_permute[i])
            circuit.append("X_ERROR", 3*N + a4_permute[i], p_prep)
    circuit.append("R", 3*N+N-1)

    circuit.append("TICK")

    for r in range(n): # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N-1:
                    circuit.append("CNOT", [a1_permute[j+i+sep], a1_permute[j+i]])
                    tick_circuit.append("CNOT", [a1_permute[j+i+sep], a1_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [a1_permute[j+i+sep], a1_permute[j+i]], p_CNOT)
                    circuit.append("CNOT", [N + a2_permute[j+i+sep], N + a2_permute[j+i]])
                    tick_circuit.append("CNOT", [N + a2_permute[j+i+sep], N + a2_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [N + a2_permute[j+i+sep], N + a2_permute[j+i]], p_CNOT)
                    circuit.append("CNOT", [2*N + a3_permute[j+i+sep], 2*N + a3_permute[j+i]])
                    tick_circuit.append("CNOT", [2*N + a3_permute[j+i+sep], 2*N + a3_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [2*N + a3_permute[j+i+sep], 2*N + a3_permute[j+i]], p_CNOT)
                    circuit.append("CNOT", [3*N + a4_permute[j+i+sep], 3*N + a4_permute[j+i]])
                    tick_circuit.append("CNOT", [3*N + a4_permute[j+i+sep], 3*N + a4_permute[j+i]])
                    circuit.append("DEPOLARIZE2", [3*N + a4_permute[j+i+sep], 3*N + a4_permute[j+i]], p_CNOT)

        circuit.append("TICK")
        tick_circuits.append(tick_circuit)


    for i in range(N-1):
        circuit.append("CNOT", [i, N+i])
        circuit.append("DEPOLARIZE2", [i, N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [i, N+i])
        circuit.append("CNOT", [2*N+i, 2*N+N+i])
        circuit.append("DEPOLARIZE2", [2*N+i, 2*N+N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [2*N+i, 2*N+N+i])
    circuit.append("TICK")
    tick_circuits.append(error_copy_circuit)

    # in experiments, here one needs to measure ancilla 2 & 4 bitwise
    # add noise to ancilla 2 & 4 here, even though they are already captured by DEPOLARIZE on CNOTs
    for i in range(N-1):
        circuit.append("X_ERROR", N+i, p_meas)
        circuit.append("X_ERROR", 3*N+i, p_meas)
    # and do classical (noisyless) processing to see if accepted
    # Stim unencode is faster than my own implementation, hence I use Stim here
    # unencode of ancilla 2 & 4 for acceptance
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [N+j+i+sep, N+j+i])    
                circuit.append("CNOT", [3*N+j+i+sep, 3*N+j+i])    

    # ancilla 2
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", N+i)
        else:
            circuit.append("M", N+i)
    circuit.append("M", N+N-1)

    # bit flip detection
    num_a2_detector = 0
    detector_str = ""
    for i in range(N):
        if bin_wt(i) < wt_thresh:
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a2_detector += 1
    # detector_str += "DETECTOR rec[-1]\n"        
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a2: {num_a2_detector}")

    # ancilla 4
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", 3*N+i)
        else:
            circuit.append("M", 3*N+i)
    circuit.append("M", 3*N+N-1)

    # bit flip detection
    num_a4_detector = 0
    detector_str = ""
    for i in range(N):
        if bin_wt(i) < wt_thresh:
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a4_detector += 1
    # detector_str += "DETECTOR rec[-1]\n"        
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a4: {num_a4_detector}")

    error_copy_circuit = stim.Circuit()
    # copy Z-error from ancilla 1 to 3
    # CNOT pointing from 3 to 1
    for i in range(N-1):
        circuit.append("CNOT", [2*N+i, i])
        circuit.append("DEPOLARIZE2", [2*N+i, i], p_CNOT)
        error_copy_circuit.append("CNOT", [2*N+i, i])
        
    tick_circuits.append(error_copy_circuit)

    # measure ancilla 3 bitwise in X-basis in experiments
    for i in range(N-1):
        circuit.append("Z_ERROR", 2*N+i, p_meas)
    # Stim processing for acceptance
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [2*N+j+i+sep, 2*N+j+i])    

    # ancilla 3
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", 2*N+i)
        else:
            circuit.append("M", 2*N+i)
    circuit.append("M", 2*N+N-1)


    # phase flip detection
    num_a3_detector = 0
    detector_str = ""
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a3_detector += 1
    # detector_str += "DETECTOR rec[-1]\n"        
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a3: {num_a3_detector}")

    # ancilla 1 detectors to see residual errors
    for r in range(n):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [j+i+sep, j+i])    
    #     circuit.append("TICK")
        
    for i in range(N-1):
        if bin_wt(i) >= wt_thresh:
            circuit.append("MX", i)
        else:
            circuit.append("M", i)
    circuit.append("M", N-1)
    num_a1_detector = 0
    detector_str = ""
    for i in range(N):
        detector_str += f"DETECTOR rec[{-N+i}]\n"
        num_a1_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a1: {num_a1_detector}")


    num_flag_detector = num_a2_detector + num_a3_detector + num_a4_detector
    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    flat_error_instructions: List[stim.DemInstruction] = [
        instruction
        for instruction in dem.flattened()
        if instruction.type == 'error'
    ]


    # start = time.time()
    # prop_dict = {}
    # print(f"total {len(flat_error_instructions)} instructions")
    # for i in range(len(flat_error_instructions)):
    #     dem_filter = stim.DetectorErrorModel()
    #     dem_filter.append(flat_error_instructions[i])
    #     explained_errors: List[stim.ExplainedError] = circuit.explain_detector_error_model_errors(dem_filter=dem_filter, reduce_to_one_representative_error=True)
    #     final_pauli_strings = []
    #     for err in explained_errors:
    #         rep_loc = err.circuit_error_locations[0]
    #         tick = rep_loc.tick_offset
    #         final_pauli_string = propagate(form_pauli_string(rep_loc.flipped_pauli_product, 4*N), tick_circuits[tick:])
    #         final_pauli_strings.append(final_pauli_string)
    #     final_pauli_product = reduce(stim.PauliString.__mul__, final_pauli_strings, stim.PauliString(4*N))
    #     final_pauli_product = final_pauli_product[:N]
    #     final_wt = final_pauli_product.weight
    #     print(f"instruction {i}, final wt on output after copying: {final_wt}. X: {final_pauli_product.pauli_indices('X')}, Y: {final_pauli_product.pauli_indices('Y')}, Z: {final_pauli_product.pauli_indices('Z')}")
    #     prop_dict[i] = final_pauli_product
    # end = time.time()
    # with open(f"{parent_dir}/propagation_dict.pkl", 'wb') as f:
    #     pickle.dump(prop_dict, f)
    # print(f"Total Elapsed time: {end-start}")   

    generation_start = time.time()
    combined_counter = Counter({}) 
    combined_one_fault_dict = Counter({})
    total_passed = 0
    fault_locations = ""
    for round in range(num_rounds):
        start = time.time()
        det_data, obs_data, err_data = dem_sampler.sample(shots=num_shots, return_errors=True, bit_packed=False)
        sample_end = time.time()
        if round == 0:
            print(f"error data shape {err_data.shape}, detector data shape {det_data.shape}", flush=True)

        not_passed = det_data[:,:num_flag_detector].any(axis=1)
        unflagged_err_data = err_data[np.logical_not(not_passed)]
        total_passed += len(unflagged_err_data)
        
        row_sums = unflagged_err_data.sum(axis=1)
        combined_counter = combined_counter + Counter(row_sums)
        one_fault_data = unflagged_err_data[row_sums == 1]
        one_fault_dict = Counter(np.nonzero(one_fault_data)[1]) # know each row only has one nonzero, extract the columns that the faults occur
        combined_one_fault_dict = combined_one_fault_dict + one_fault_dict

        for single_shot_err_data in unflagged_err_data[row_sums >= 2]:
            fault_locations += str(np.nonzero(single_shot_err_data)[0]) + "\n"
            to_print = ""
            num_faults = np.count_nonzero(single_shot_err_data)
            dem_filter = stim.DetectorErrorModel()
            for error_index in np.flatnonzero(single_shot_err_data):
                dem_filter.append(flat_error_instructions[error_index])
            explained_errors: List[stim.ExplainedError] = circuit.explain_detector_error_model_errors(dem_filter=dem_filter, reduce_to_one_representative_error=True)
            ticks_after_prep = [err.circuit_error_locations[0].tick_offset >= 8 for err in explained_errors]
            if all(ticks_after_prep): continue # error happened on copying CNOT gates
            to_print += f"{num_faults} faults occurred\n"
            final_pauli_strings = []
            for err in explained_errors:
                rep_loc = err.circuit_error_locations[0]
                to_print += f"{rep_loc}\n"
                tick = rep_loc.tick_offset
                final_pauli_string = propagate(form_pauli_string(rep_loc.flipped_pauli_product, 4*N), tick_circuits[tick:])
                final_pauli_strings.append(final_pauli_string)
                to_print += f"fault at tick {tick}, {rep_loc.flipped_pauli_product}, final wt: {final_pauli_string.weight}. X: {final_pauli_string.pauli_indices('X')}, Y: {final_pauli_string.pauli_indices('Y')}, Z: {final_pauli_string.pauli_indices('Z')}\n"
            final_pauli_product = reduce(stim.PauliString.__mul__, final_pauli_strings, stim.PauliString(4*N))
            final_wt = final_pauli_product.weight
            to_print += f"final wt after copying: {final_wt}. X: {final_pauli_product.pauli_indices('X')}, Y: {final_pauli_product.pauli_indices('Y')}, Z: {final_pauli_product.pauli_indices('Z')}\n"
            final_pauli_product = final_pauli_product[:N]
            final_wt = final_pauli_product.weight
            if final_wt >= num_faults:
                to_print += f"final wt on output after copying: {final_wt}. X: {final_pauli_product.pauli_indices('X')}, Y: {final_pauli_product.pauli_indices('Y')}, Z: {final_pauli_product.pauli_indices('Z')}"
                print(to_print, flush=True)

        end = time.time()
        if round == 0:
            print(f"Stim sampling elapsed time per {num_shots} samples: {sample_end-start} second, with postprocessing {end-start}", flush=True)
        if (round+1) % 10 == 0: # print every 1e6 samples
            print("Temporary counter for among all passed samples, how many faults occured:", combined_counter, flush=True)
        
    print(f"Among {num_rounds * num_shots} samples, {total_passed} passed.")
    print("Counter for among all passed samples, how many faults occured:", combined_counter, flush=True)
    print("Number of passing one fault location:", len(combined_one_fault_dict), flush=True)
    print(f"Total elaspsed time: {time.time() - generation_start} seconds", flush=True)
    with open(f"{parent_dir}/{input_value}_faults.log", 'w') as f:
        f.write(fault_locations)
    with open(f"{parent_dir}/{input_value}_single_fault.pkl", 'wb') as f:
        pickle.dump(combined_one_fault_dict, f)
