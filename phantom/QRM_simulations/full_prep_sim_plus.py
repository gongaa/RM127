import stim
print(stim.__version__)
import numpy as np
from typing import List
import time
from collections import Counter
from functools import reduce
import sys
import pickle
from utils import propagate, form_pauli_string
import settings 
from settings import int2bin, bin2int, bin_wt 
sys.path.append("../")
from FT_prep.PyDecoder_polar import PyDecoder_polar_SCL

m = settings.m
N = settings.N
l = settings.l
K = settings.K
d = settings.d

# phantom QRM parameter [[2^m, m-l+1, 2^{m-l} / 2^l]]
print(f"full_prep_sim_plus.py: m={m}, N={N}, K={K}, d={d}")
state = '+' # prepare the all-plus state
initial_state = ['0'] * N
for i in range(N):
    if bin_wt(N-1-i) < l:
        initial_state[i] = "+"
# prepare the all-plus state
for i in range(l-1, m):
    idx = list("1"*(l-1)+"0"*(m-l+1))
    idx[i] = "1"
    initial_state[N-1-bin2int("".join(idx)[::-1])] = "+"

A1 = np.eye(m, dtype=int)
A2 = np.array( [[1, 0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1, 0],
                [1, 1, 0, 1, 1, 1],
                [1, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 0, 0, 1]])
A3 = np.array( [[0, 1, 1, 1, 0, 0],
                [1, 1, 0, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1, 1]])
A4 = np.array( [[1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 1, 1, 0, 0, 1],
                [1, 1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 1]])

Ax = lambda A, i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
a1_permute = [Ax(A1, i) for i in range(N)]
a2_permute = [Ax(A2, i) for i in range(N)]
a3_permute = [Ax(A3, i) for i in range(N)]
a4_permute = [Ax(A4, i) for i in range(N)]

decoder = PyDecoder_polar_SCL(l, m, 1)

if __name__ == "__main__":
    # Check if an argument has been provided
    if len(sys.argv) != 3:
        print("Usage: python script.py index<int> p_CNOT<float>")
        sys.exit(1)
    try:
        # Get the integer from the command line argument
        input_value = int(sys.argv[1])
        error_rate = float(sys.argv[2])
    except ValueError:
        print("The argument must be an integer.")
        sys.exit(1)
        
    p_CNOT = error_rate
    p_meas = 5.0 * p_CNOT / 3.0
    p_reset = p_CNOT
    p_idle = p_CNOT / 300.0
    p_idle_perm = p_CNOT / 6.0
    num_rounds = 1000
    num_shots = 10000
    parent_dir = "logs_prep_plus"
    parent_dir += '/p' + str(p_CNOT).split('.')[1] # comment this line out if generating propagation_dict
    print(f"full_prep_sim_plus.py writing to {parent_dir}/{input_value}.log", flush=True)
    print(f"p_CNOT={p_CNOT}, p_measure={p_meas}, p_reset={p_reset}")

    circuit = stim.Circuit()
    error_copy_circuit = stim.Circuit()

    tick_circuits = [] # for PauliString.after


    # ancilla 1
    for i in range(N):
        if initial_state[i] == '+':
            circuit.append("RX", a1_permute[i])
            circuit.append("Z_ERROR", a1_permute[i], p_reset)
        else:
            circuit.append("R", a1_permute[i])
            circuit.append("X_ERROR", a1_permute[i], p_reset)

    # ancilla 2
    for i in range(N):
        if initial_state[i] == '+':
            circuit.append("RX", N + a2_permute[i])
            circuit.append("Z_ERROR", N + a2_permute[i], p_reset)
        else:
            circuit.append("R", N + a2_permute[i])
            circuit.append("X_ERROR", N + a2_permute[i], p_reset)

    # ancilla 3
    for i in range(N):
        if initial_state[i] == '+':
            circuit.append("RX", 2*N + a3_permute[i])
            circuit.append("Z_ERROR", 2*N + a3_permute[i], p_reset)
        else:
            circuit.append("R", 2*N + a3_permute[i])
            circuit.append("X_ERROR", 2*N + a3_permute[i], p_reset)

    # ancilla 4
    for i in range(N):
        if initial_state[i] == '+':
            circuit.append("RX", 3*N + a4_permute[i])
            circuit.append("Z_ERROR", 3*N + a4_permute[i], p_reset)
        else:
            circuit.append("R", 3*N + a4_permute[i])
            circuit.append("X_ERROR", 3*N + a4_permute[i], p_reset)

    circuit.append("TICK")

    for r in range(m): # rounds
        sep = 2 ** r
        tick_circuit = stim.Circuit()
        for j in range(0, N, 2*sep):
            for i in range(sep):
                if j+i+sep < N:
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

    # four patches idling during permutation
    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle_perm)
        circuit.append("DEPOLARIZE1", N+i, p_idle_perm)
        circuit.append("DEPOLARIZE1", 2*N+i, p_idle_perm)
        circuit.append("DEPOLARIZE1", 3*N+i, p_idle_perm)

    # Z error detection first
    # copy Z error from ancilla 1 to 2, and 3 to 4, then measure 2 in X basis, 4 in X basis
    for i in range(N):
        circuit.append("CNOT", [N+i, i])
        circuit.append("DEPOLARIZE2", [N+i, i], p_CNOT)
        error_copy_circuit.append("CNOT", [N+i, i])
        circuit.append("CNOT", [2*N+N+i, 2*N+i])
        circuit.append("DEPOLARIZE2", [2*N+N+i, 2*N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [2*N+N+i, 2*N+i])
    circuit.append("TICK")
    tick_circuits.append(error_copy_circuit)

    # in experiments, here one needs to measure ancilla 2 & 4 bitwise
    # add measurement noise to ancilla 2 & 4 here
    for i in range(N):
        circuit.append("Z_ERROR", N+i, p_meas)
        circuit.append("Z_ERROR", 3*N+i, p_meas)
    # and do classical (noisyless) processing to see if accepted
    # Stim unencode is faster than my own implementation, hence I use Stim here
    # unencode of ancilla 2 & 4 for acceptance
    for r in range(m):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [N+j+i+sep, N+j+i])    
                circuit.append("CNOT", [3*N+j+i+sep, 3*N+j+i])     

    # ancilla 2 phase flip detection
    for i in range(N):
        if initial_state[i] == "+":
            circuit.append("MX", N+i)
        else:
            circuit.append("M", N+i)

    num_a2_detector = 0
    detector_str = ""
    for i in range(N):
        if initial_state[i] == "+":
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a2_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a2: {num_a2_detector}")

    # ancilla 4 phase flip detection
    for i in range(N):
        if initial_state[i] == "+":
            circuit.append("MX", 3*N+i)
        else:
            circuit.append("M", 3*N+i)

    num_a4_detector = 0
    detector_str = ""
    for i in range(N):
        if initial_state[i] == "+":
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a4_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a4: {num_a4_detector}")


    error_copy_circuit = stim.Circuit()
    # copy X-error from ancilla 1 to 3, then measure 3 in Z basis
    # CNOT pointing from 1 to 3
    for i in range(N):
        circuit.append("CNOT", [i, 2*N+i])
        circuit.append("DEPOLARIZE2", [i, 2*N+i], p_CNOT)
        error_copy_circuit.append("CNOT", [i, 2*N+i])
        
    tick_circuits.append(error_copy_circuit)

    # measure ancilla 3 bitwise in Z-basis in experiments
    for i in range(N):
        circuit.append("X_ERROR", 2*N+i, p_meas)
    
    # ancilla 1 idling during ancilla 3 measurement
    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)

    # Stim processing for acceptance on ancilla 3, bit flip detection
    for r in range(m):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [2*N+j+i+sep, 2*N+j+i])      

    for i in range(N):
        if initial_state[i] == "+":
            circuit.append("MX", 2*N+i)
        else:
            circuit.append("M", 2*N+i)

    num_a3_detector = 0
    detector_str = ""
    for i in range(N):
        if initial_state[i] == "0":
            detector_str += f"DETECTOR rec[{-N+i}]\n"
            num_a3_detector += 1
    detector_circuit = stim.Circuit(detector_str)
    circuit += detector_circuit
    print(f"#detectors put on a3: {num_a3_detector}")

    # ancilla 1 detectors to see residual errors
    for r in range(m):
        sep = 2 ** r
        for j in range(0, N, 2*sep):
            for i in range(sep):
                circuit.append("CNOT", [j+i+sep, j+i])    
        
    for i in range(N):
        if initial_state[i] == "+":
            circuit.append("MX", i)
        else:
            circuit.append("M", i)
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

    # Uncomment the following to generate propagation dictionary.
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
    # exit()

    # State preparation simulation. Comment them out when generating propagation dictionary.
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
            if final_wt > num_faults:
                to_print += f"final wt on output after copying: {final_wt}. X: {final_pauli_product.pauli_indices('X')}, Y: {final_pauli_product.pauli_indices('Y')}, Z: {final_pauli_product.pauli_indices('Z')}"
                x, z = final_pauli_product.to_numpy()
                Z_residual_error_wt = decoder.decode_Z_flip(list(np.where(z)[0]))
                X_residual_error_wt = decoder.decode_X_flip(list(np.where(x)[0]))
                to_print += f"\nX residual error wt {X_residual_error_wt}, Z residual error wt {Z_residual_error_wt}"
                print(to_print, flush=True)

        end = time.time()
        if round == 0:
            print(f"Stim sampling elapsed time per {num_shots} samples: {sample_end-start} second, with postprocessing {end-start}", flush=True)
        if (round+1) % 10 == 0:
            print("Temporary counter for among all passed samples, how many faults occured:", combined_counter, flush=True)
        
    print(f"Among {num_rounds * num_shots} samples, {total_passed} passed.")
    print("Counter for among all passed samples, how many faults occured:", combined_counter, flush=True)
    print("Number of passing one fault location:", len(combined_one_fault_dict), flush=True)
    print(f"Total elaspsed time: {time.time() - generation_start} seconds", flush=True)
    with open(f"{parent_dir}/{input_value}_faults.log", 'w') as f:
        f.write(fault_locations)
    with open(f"{parent_dir}/{input_value}_single_fault.pkl", 'wb') as f:
        pickle.dump(combined_one_fault_dict, f)