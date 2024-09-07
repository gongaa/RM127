import stim
print(stim.__version__)
import numpy as np
import time, sys, argparse
from PyDecoder_polar import PyDecoder_polar_SCL
from utils import z_component, x_component, sample_ancilla_error, process_ancilla_error

N = 2 ** 7
factor = 1.0            # p_SPAM / p_CNOT
factor_single = 1.0     # p_single / p_CNOT
factor_correction = 1.0 # p_correction / p_CNOT, will use 0.0 for Clifford gate (Pauli Frame Tracking)
p_CNOT = 0.001
p_SPAM = factor * p_CNOT # state preparation and measurement
p_single = factor_single * p_CNOT # single qubit gate, H, S, T, transversal on all qubits
p_correction = factor_correction * p_CNOT # single qubit addressing, arbitrary Pauli string
dir_error_rate = ""
bs = 1024
def mean_wt(errors):
    x_wt = [len(s.pauli_indices('XY')) for s in errors]
    x = [s.pauli_indices('XY') for s in errors]
    z_wt = [len(s.pauli_indices('ZY')) for s in errors]
    z = [s.pauli_indices('YZ') for s in errors]
    y_wt = [len(s.pauli_indices('Y')) for s in errors]
    y = [s.pauli_indices('Y') for s in errors]
    print(f"mean XY wt: {sum(x_wt)/bs}, mean YZ wt: {sum(z_wt)/bs}, mean pure Y component: {sum(y_wt)/bs}")
    # print(f"XY component {x}, YZ component {z}, pure Y component {y}")

# phase error Steane EC block
def phase_flip_EC_block(ancilla_errors, input_errors=None, decoder=None, disable_correction_error=False, avg_flip=None, alert=False):
    global p_CNOT, p_SPAM, p_correction
    if input_errors is None:
        input_errors = [stim.PauliString(N) for _ in range(bs)]
    # broadcast incoming noise to qubit 0 ~ N-1
    # broadcast ancilla noise to qubit N ~ 2N-1
    # print("phase EC block input noise")
    # mean_wt(input_errors)
    before_coupling_errors = [e1+e2 for (e1,e2) in zip(input_errors, ancilla_errors)] # list of length 2N PauliString
    
    coupling_circuit = stim.Circuit()
    if not disable_correction_error: 
        # for example: if pauli-frame tracking is not turned on
        # or for example: first bit-flip then phase-flip EC, 
        # then correction can be applied when both blocks are done, 
        # sparing the correction error for the second block
        for i in range(N-1): # when applying previous round error correction operator, there is single-qubit noise
            coupling_circuit.append("DEPOLARIZE1", i, p_correction)
    for i in range(N-1):
        coupling_circuit.append("CNOT", [i+N, i])
        coupling_circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)

    for i in range(N-1): # measurement noise
        coupling_circuit.append("Z_ERROR", N+i, p_SPAM)
    coupling_sim = stim.FlipSimulator(batch_size=bs, num_qubits=2*N, disable_stabilizer_randomization=True)
    
    X_component, Z_component = np.array([e.to_numpy() for e in before_coupling_errors]).transpose(1,2,0) # each shape (2*N, bs)
    coupling_sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    coupling_sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    coupling_sim.do(coupling_circuit)
    coupling_errors = coupling_sim.peek_pauli_flips()

    if decoder is not None:
        # look at Z component on qubit N ~ 2N-1 and try to correct
        noise = list(map(lambda s: z_component(s[N:]), coupling_errors))
        # if alert:
        #     print("before applying phase-flip decoder")
        #     mean_wt(noise)
        if avg_flip is not None:
            avg_flip.append(sum([len(s.pauli_indices('Z')) for s in noise])/bs) # same as 'YZ' for z component

        residual_errors = list(map(lambda s: s[:N] * z_component(s[N:]), coupling_errors))
        error_mask = [False for _ in range(bs)]
        for i in range(len(noise)):
            num_flip = decoder.decode(list(np.nonzero(noise[i])[0]))
            class_bit = decoder.last_info_bit
            if class_bit != 0:
                error_mask[i] = True
                if alert:
                    print(f"ALERT: before logical gates correction error, apply an extra logical Z. Before decoder noise={np.nonzero(noise[i])[0]}. After decoder #flip={num_flip}, correction={decoder.correction}", flush=True)
                    residual_errors[i] *= stim.PauliString("Z"*(N-1)+"I")
                # print("# phase flip:", num_flip)
        # print("after a phase EC block with decoder")
        # mean_wt(residual_errors)
        return residual_errors, np.array(error_mask, dtype=np.bool_)
    else: # assume perfect correction first
        residual_errors = list(map(lambda s: s[:N] * z_component(s[N:]), coupling_errors))
        # print("after a phase EC block")
        # mean_wt(residual_errors)
        return residual_errors

def bit_flip_EC_block(ancilla_errors, input_errors=None, decoder=None, disable_correction_error=False, avg_flip=[], alert=False):
    global p_CNOT, p_SPAM, p_correction
    if input_errors is None:
        input_errors = [stim.PauliString(N) for _ in range(bs)]
    # broadcast incoming noise to qubit 0 ~ N-1
    # broadcast ancilla noise to qubit N ~ 2N-1
    # print("bit EC block input noise")
    # mean_wt(input_errors)
    before_coupling_errors = [e1+e2 for (e1,e2) in zip(input_errors, ancilla_errors)] # list of length 2N PauliString
    
    coupling_circuit = stim.Circuit()
    if not disable_correction_error: 
        for i in range(N-1): # previous round EC operation noise
            coupling_circuit.append("DEPOLARIZE1", i, p_correction)
    for i in range(N-1):
        coupling_circuit.append("CNOT", [i, i+N])
        coupling_circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)

    for i in range(N-1): # measurement noise
        coupling_circuit.append("X_ERROR", N+i, p_SPAM)
    coupling_sim = stim.FlipSimulator(batch_size=bs, num_qubits=2*N, disable_stabilizer_randomization=True)
    
    X_component, Z_component = np.array([e.to_numpy() for e in before_coupling_errors]).transpose(1,2,0) # each shape (2*N, bs)
    coupling_sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    coupling_sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    coupling_sim.do(coupling_circuit)
    coupling_errors = coupling_sim.peek_pauli_flips()

    if decoder is not None:
        # look at X component on qubit N ~ 2N-1 and try to correct
        noise = list(map(lambda s: x_component(s[N:]), coupling_errors))
        # if alert:
        #     print("before applying bit-flip decoder")
        #     mean_wt(noise)
        if avg_flip is not None:
            avg_flip.append(sum([len(s.pauli_indices('X')) for s in noise])/bs)

        residual_errors = list(map(lambda s: s[:N] * x_component(s[N:]), coupling_errors))
        error_mask = [False for _ in range(bs)]
        for i in range(len(noise)):
            num_flip = decoder.decode(list(np.nonzero(noise[i])[0]))
            class_bit = decoder.last_info_bit
            if class_bit != 0:
                error_mask[i] = True
                if alert:
                    print(f"ALERT: before logical gates correction error, apply an extra logical X. Before decoder noise={np.nonzero(noise[i])[0]}. After decoder #flip={num_flip}, correction={decoder.correction}", flush=True)
                    residual_errors[i] *= stim.PauliString("X"*(N-1)+"I")

                # print("# bit flip:", num_flip)
        # print("after a bit EC block with decoder")
        # mean_wt(residual_errors)
        return residual_errors, np.array(error_mask, dtype=np.bool_)
    else: # assume perfect correction first
        residual_errors = list(map(lambda s: s[:N] * x_component(s[N:]), coupling_errors))
        # print("after a bit EC block")
        # mean_wt(residual_errors)
        return residual_errors


def logical_single_qubit_Clifford(input_noise, gate_type='H'): # transversal Hadamard or S gate, without leading/trailing EC 
    global p_CNOT, p_SPAM, p_single, p_correction
    if gate_type not in ['H', 'S']:
        print("Unsupported gate type")
        return
    transversal_logical_circuit = stim.Circuit()
    for i in range(N-1): # previous round EC operation noise
        transversal_logical_circuit.append("DEPOLARIZE1", i, p_correction)
    for i in range(N-1): 
        transversal_logical_circuit.append(gate_type, i)
        transversal_logical_circuit.append("DEPOLARIZE1", i, p_single)

    logical_sim = stim.FlipSimulator(batch_size=bs, num_qubits=N, disable_stabilizer_randomization=True)
    X_component, Z_component = np.array([e.to_numpy() for e in input_noise]).transpose(1,2,0) # each shape (N, bs)
    logical_sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    logical_sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    logical_sim.do(transversal_logical_circuit)
    residual_errors = logical_sim.peek_pauli_flips()
    return residual_errors

def logical_CNOT(input_noise_control, input_noise_target): # transversal CNOT, without leading/trailing EC 
    global p_CNOT, p_SPAM, p_correction
    transversal_CNOT_circuit = stim.Circuit()
    for i in range(N-1): # previous round EC operation noise
        transversal_CNOT_circuit.append("DEPOLARIZE1", i, p_correction)
        transversal_CNOT_circuit.append("DEPOLARIZE1", i+N, p_correction)
    for i in range(N-1): 
        transversal_CNOT_circuit.append("CNOT", [i, i+N])
        transversal_CNOT_circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)

    before_errors = [e1+e2 for (e1,e2) in zip(input_noise_control, input_noise_target)] # list of length 2N PauliString
    logical_sim = stim.FlipSimulator(batch_size=bs, num_qubits=2*N, disable_stabilizer_randomization=True)
    X_component, Z_component = np.array([e.to_numpy() for e in before_errors]).transpose(1,2,0) # each shape (N, bs)
    logical_sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    logical_sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    logical_sim.do(transversal_CNOT_circuit)
    residual_errors = logical_sim.peek_pauli_flips()
    return [e[:N] for e in residual_errors], [e[N:] for e in residual_errors] # residual errors on control and target
    
def logical_T(input_noise): # transversal T gate
    # Z error remains Z error
    # X stays X with prob. 1/2, becomes Y with prob. 1/2
    # Y stays Y with prob. 1/2, becomes X with prob. 1/2
    # i.e., X remains X, while also has 1/2 prob. adding to Z
    global p_CNOT, p_SPAM, p_single, p_correction
    single_qubit_correction_noise_circuit = stim.Circuit()
    for i in range(N-1): # previous round EC operation noise
        single_qubit_correction_noise_circuit.append("DEPOLARIZE1", i, p_correction)
    logical_sim = stim.FlipSimulator(batch_size=bs, num_qubits=N, disable_stabilizer_randomization=True)
    X_component, Z_component = np.array([e.to_numpy() for e in input_noise]).transpose(1,2,0) # each shape (N, bs)
    logical_sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    logical_sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    logical_sim.do(single_qubit_correction_noise_circuit)
    before_T_errors = logical_sim.peek_pauli_flips()
    X_component = np.array([e.to_numpy()[0] for e in before_T_errors]) # shape (bs, N)
    random_mask = np.random.rand(bs, N) < 0.5
    changed_component = X_component & random_mask
    logical_sim.broadcast_pauli_errors(pauli='Z', mask=changed_component.T)

    single_qubit_noise_circuit = stim.Circuit() # transversal T gate noise
    for i in range(N-1): # previous round EC operation noise
        single_qubit_noise_circuit.append("DEPOLARIZE1", i, p_single)
    logical_sim.do(single_qubit_noise_circuit)
    residual_errors = logical_sim.peek_pauli_flips()
    return residual_errors
    

def simulate_code_switching_rectangle(num_batch=1000, index=0):    # batch size fixed to 1024
    global dir_error_rate, factor
    decoder_r4 = PyDecoder_polar_SCL(4)
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_Z_errors = 0; total_X_errors = 0 # want to confirm the phase flip is the dominant term
    num_shots = num_batch * bs
    start = time.time()
    a1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 2*index, dir_error_rate, factor)
    a2_d7_plus = sample_ancilla_error(num_shots, 7, 'plus', index, dir_error_rate, factor)
    a3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 2*index+1, dir_error_rate, factor)
    a4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', index, dir_error_rate, factor)
    avg_Z_flip = []
    avg_X_flip = []
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1 = phase_flip_EC_block(ancilla_errors=a1_d15_zero[s_start:s_end])
        residual_errors_b2 = bit_flip_EC_block(input_errors=residual_errors_b1, ancilla_errors=a2_d7_plus[s_start:s_end], disable_correction_error=True)
        # TODO: double-check logical T gate 
        residual_errors = logical_T(residual_errors_b2)
        residual_errors_b3, error_mask_phase = phase_flip_EC_block(input_errors=residual_errors, ancilla_errors=a3_d15_zero[s_start:s_end], decoder=decoder_r4, disable_correction_error=True, avg_flip=avg_Z_flip)
        residual_errors_b4, error_mask_bit = bit_flip_EC_block(input_errors=residual_errors_b3, ancilla_errors=a4_d15_plus[s_start:s_end], decoder=decoder_r3, avg_flip=avg_X_flip)
        total_num_errors += np.logical_or(error_mask_phase, error_mask_bit).astype(int).sum()
        total_Z_errors += sum(error_mask_phase)
        total_X_errors += sum(error_mask_bit)
        print(f"#errors/#samples: {total_num_errors}/{s_end}")
        print(f"#phase errors/#samples: {total_Z_errors}/{s_end}, #bit errors/#samples: {total_X_errors}/{s_end}")
        print(f"error rate: {total_num_errors/(s_end)}")
        print(f"average Z flips: {sum(avg_Z_flip)/(round+1)}, average X flips: {sum(avg_X_flip)/(round+1)}")
    end = time.time()
    print(f"Total elasped time {end-start} seconds.")

def simulate_code_switching_rectangle_special_load(num_batch=1000, index=0):    # batch size fixed to 1024
    global dir_error_rate, factor
    decoder_r4 = PyDecoder_polar_SCL(4)
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_Z_errors = 0; total_X_errors = 0 # want to confirm the phase flip is the dominant term
    num_shots = num_batch * bs
    import pickle
    if factor == 1.0:
        with open(f"logs_prep_SPAM_equal_CNOT/d15_zero/propagation_dict.pkl", 'rb') as f:
            d15_zero_prop_dict = pickle.load(f)
        with open(f"logs_prep_SPAM_equal_CNOT/d15_plus/propagation_dict.pkl", 'rb') as f:
            d15_plus_prop_dict = pickle.load(f)
        with open(f"logs_prep_SPAM_equal_CNOT/d7_plus/propagation_dict.pkl", 'rb') as f:
            d7_plus_prop_dict = pickle.load(f)
    else:
        with open(f"logs_prep_SPAM_half_CNOT/d15_zero/propagation_dict.pkl", 'rb') as f:
            d15_zero_prop_dict = pickle.load(f)
        with open(f"logs_prep_SPAM_half_CNOT/d15_plus/propagation_dict.pkl", 'rb') as f:
            d15_plus_prop_dict = pickle.load(f)
        with open(f"logs_prep_SPAM_half_CNOT/d7_plus/propagation_dict.pkl", 'rb') as f:
            d7_plus_prop_dict = pickle.load(f)

    start = time.time()
    a1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 2*index, dir_error_rate, factor, postprocess=False)
    a2_d7_plus = sample_ancilla_error(num_shots, 7, 'plus', index, dir_error_rate, factor, postprocess=False)
    a3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 2*index+1, dir_error_rate, factor, postprocess=False)
    a4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', index, dir_error_rate, factor, postprocess=False)
    avg_Z_flip = []
    avg_X_flip = []
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1 = phase_flip_EC_block(ancilla_errors=process_ancilla_error(d15_zero_prop_dict, a1_d15_zero[s_start:s_end]))
        residual_errors_b2 = bit_flip_EC_block(input_errors=residual_errors_b1, ancilla_errors=process_ancilla_error(d7_plus_prop_dict, a2_d7_plus[s_start:s_end]), disable_correction_error=True)
        # TODO: double-check logical T gate 
        residual_errors = logical_T(residual_errors_b2)
        residual_errors_b3, error_mask_phase = phase_flip_EC_block(input_errors=residual_errors, ancilla_errors=process_ancilla_error(d15_zero_prop_dict, a3_d15_zero[s_start:s_end]), decoder=decoder_r4, disable_correction_error=True, avg_flip=avg_Z_flip)
        residual_errors_b4, error_mask_bit = bit_flip_EC_block(input_errors=residual_errors_b3, ancilla_errors=process_ancilla_error(d15_plus_prop_dict, a4_d15_plus[s_start:s_end]), decoder=decoder_r3, avg_flip=avg_X_flip)
        total_num_errors += np.logical_or(error_mask_phase, error_mask_bit).astype(int).sum()
        total_Z_errors += sum(error_mask_phase)
        total_X_errors += sum(error_mask_bit)
        print(f"#errors/#samples: {total_num_errors}/{s_end}")
        print(f"#phase errors/#samples: {total_Z_errors}/{s_end}, #bit errors/#samples: {total_X_errors}/{s_end}")
        print(f"error rate: {total_num_errors/(s_end)}")
        print(f"average Z flips: {sum(avg_Z_flip)/(round+1)}, average X flips: {sum(avg_X_flip)/(round+1)}")
    end = time.time()
    print(f"Total elasped time {end-start} seconds.")



def simulate_single_qubit_Clifford_rectangle(num_batch=150, gate_type='H'): # extended rectangle for logical transversal H or S
    if gate_type not in ['H', 'S']:
        print("Unsupported gate type")
        return
    global dir_error_rate, factor
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_Z_errors = 0; total_X_errors = 0 # want to confirm the phase flip is the dominant term
    num_shots = num_batch * bs
    start = time.time()
    a1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 0, dir_error_rate, factor)
    a2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 0, dir_error_rate, factor)
    a3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 1, dir_error_rate, factor)
    a4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 1, dir_error_rate, factor)
    avg_Z_flip = []
    avg_X_flip = []
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1 = phase_flip_EC_block(ancilla_errors=a1_d15_zero[s_start:s_end], decoder=decoder_r3, alert=True)
        residual_errors_b2 = bit_flip_EC_block(input_errors=residual_errors_b1, ancilla_errors=a2_d15_plus[s_start:s_end], decoder=decoder_r3, alert=True)
        residual_errors = logical_single_qubit_Clifford(residual_errors_b2, gate_type)
        residual_errors_b3, error_mask_phase = phase_flip_EC_block(input_errors=residual_errors, ancilla_errors=a3_d15_zero[s_start:s_end], decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_Z_flip)
        residual_errors_b4, error_mask_bit = bit_flip_EC_block(input_errors=residual_errors_b3, ancilla_errors=a4_d15_plus[s_start:s_end], decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_X_flip)
        total_num_errors += np.logical_or(error_mask_phase, error_mask_bit).astype(int).sum()
        total_Z_errors += sum(error_mask_phase)
        total_X_errors += sum(error_mask_bit)
        print(f"#errors/#samples: {total_num_errors}/{s_end}")
        print(f"#phase errors/#samples: {total_Z_errors}/{s_end}, #bit errors/#samples: {total_X_errors}/{s_end}")
        print(f"error rate: {total_num_errors/(s_end)}")
        print(f"average Z flips: {sum(avg_Z_flip)/(round+1)}, average X flips: {sum(avg_X_flip)/(round+1)}")
    end = time.time()
    print(f"Total elasped time {end-start} seconds.")

def simulate_CNOT_rectangle(num_batch=150, index=0): # extended rectangle for logical transversal CNOT between two blocks
    # correct phase-flip first, then bit-flip
    # target bit-flip > control phase-flip. IX > ZI
    ''' -------X--------------.------corr----.----------X--------------.-------
    CONTROL    |              |       ||     |          |              |
       ca1 |0> .--MX  ca2 |+> X--MZ ===      |  ca3 |0> .--MX  ca4 |+> X--MZ
                                             |
        -------X--------------.------corr----X----------X--------------.-------
    TARGET     |              |       ||                |              |
       ta1 |0> .--MX  ta1 |+> X--MZ ===         ta3 |0> .--MX  ta4 |+> X--MZ
    '''
    global dir_error_rate, factor, bs
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_before_logical_errors = 0
    total_num_errors_control = 0; total_num_errors_target = 0
    num_shots = num_batch * bs
    start = time.time()
    ca1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index, dir_error_rate, factor)
    ca2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index, dir_error_rate, factor)
    ca3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index+1, dir_error_rate, factor)
    ca4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index+1, dir_error_rate, factor)
    ta1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index+2, dir_error_rate, factor)
    ta2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index+2, dir_error_rate, factor)
    ta3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index+3, dir_error_rate, factor)
    ta4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index+3, dir_error_rate, factor)
    avg_c_Z_flip, avg_t_Z_flip = [], []
    avg_c_X_flip, avg_t_X_flip = [], []
    num_ZI, num_IZ, num_ZZ, num_IX, num_XI, num_XX = 0,0,0,0,0,0
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1_control, error_mask_before_log_phase_control = phase_flip_EC_block(ancilla_errors=ca1_d15_zero[s_start:s_end], decoder=decoder_r3, alert=True) # TODO: expect same LER when turn disable_correction_error=True
        residual_errors_b2_control, error_mask_before_log_bit_control = bit_flip_EC_block(input_errors=residual_errors_b1_control, ancilla_errors=ca2_d15_plus[s_start:s_end], disable_correction_error=True, decoder=decoder_r3, alert=True)
        residual_errors_b1_target, error_mask_before_log_phase_target = phase_flip_EC_block(ancilla_errors=ta1_d15_zero[s_start:s_end], decoder=decoder_r3, alert=True) # TODO: disable_correction_error=True
        residual_errors_b2_target, error_mask_before_log_bit_target = bit_flip_EC_block(input_errors=residual_errors_b1_target, ancilla_errors=ta2_d15_plus[s_start:s_end], disable_correction_error=True, decoder=decoder_r3, alert=True)
        residual_errors_control, residual_errors_target = logical_CNOT(residual_errors_b2_control, residual_errors_b2_target)
        residual_errors_b3_control, error_mask_phase_control = phase_flip_EC_block(input_errors=residual_errors_control, ancilla_errors=ca3_d15_zero[s_start:s_end], decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_c_Z_flip) # preceding block is CNOT, not LEC
        residual_errors_b4_control, error_mask_bit_control = bit_flip_EC_block(input_errors=residual_errors_b3_control, ancilla_errors=ca4_d15_plus[s_start:s_end], decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_c_X_flip)
        residual_errors_b3_target, error_mask_phase_target = phase_flip_EC_block(input_errors=residual_errors_target, ancilla_errors=ta3_d15_zero[s_start:s_end], decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_t_Z_flip)
        residual_errors_b4_target, error_mask_bit_target = bit_flip_EC_block(input_errors=residual_errors_b3_target, ancilla_errors=ta4_d15_plus[s_start:s_end], decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_t_X_flip)
        control_errors = np.logical_or(error_mask_phase_control, error_mask_bit_control)
        target_errors = np.logical_or(error_mask_phase_target, error_mask_bit_target)
        total_after_logical_error = np.logical_or(control_errors, target_errors)
        total_num_errors += total_after_logical_error.astype(int).sum()
        total_before_logical_error = np.logical_or(np.logical_or(error_mask_before_log_phase_control, error_mask_before_log_bit_control), np.logical_or(error_mask_before_log_phase_target, error_mask_before_log_bit_target))
        total_before_logical_errors += total_before_logical_error.astype(int).sum()
        total_num_errors_control += sum(control_errors)
        total_num_errors_target += sum(target_errors)
        num_ZI += sum(error_mask_phase_control)
        num_XI += sum(error_mask_bit_control)
        num_IZ += sum(error_mask_phase_target)
        num_IX += sum(error_mask_bit_target)
        num_ZZ += sum(np.logical_and(error_mask_phase_control, error_mask_phase_target))
        num_XX += sum(np.logical_and(error_mask_bit_control, error_mask_bit_target))
        if total_before_logical_error.any():
            print(f"before logical error happened on sample {np.where(total_before_logical_error)[0]} within batch")
            print(f"after logical error happened on sample {np.where(total_after_logical_error)[0]} within batch", flush=True)
    end = time.time()
    print(f"#errors before logical/#samples: {total_before_logical_errors}/{s_end}")
    print(f"#errors/#samples: {total_num_errors}/{s_end}")
    # print(f"#control errors/#samples: {total_num_errors_control}/{s_end}, #phase flip: {sum(error_mask_phase_control)}, #bit flip: {sum(error_mask_bit_control)}")
    # print(f"#target errors/#samples: {total_num_errors_target}/{s_end}, #phase flip: {sum(error_mask_phase_target)}, #bit flip: {sum(error_mask_bit_target)}")
    print(f"ZI: {num_ZI}, XI: {num_XI}, IZ: {num_IZ}, IX: {num_IX}, ZZ: {num_ZZ}, XX: {num_XX}")
    print(f"error rate: {total_num_errors/(s_end)}")
    print(f"control average Z flips: {sum(avg_c_Z_flip)/(round+1)}, X flips: {sum(avg_c_X_flip)/(round+1)}, target average Z flips: {sum(avg_t_Z_flip)/(round+1)}, X flips: {sum(avg_t_X_flip)/(round+1)}")
    print(f"Total elasped time {end-start} seconds.", flush=True)

def simulate_CNOT_rectangle_special_load(num_batch=150, index=0): # extended rectangle for logical transversal CNOT between two blocks
    # correct phase-flip first, then bit-flip
    # target bit-flip > control phase-flip. IX > ZI
    ''' -------X--------------.------corr----.----------X--------------.-------
    CONTROL    |              |       ||     |          |              |
       ca1 |0> .--MX  ca2 |+> X--MZ ===      |  ca3 |0> .--MX  ca4 |+> X--MZ
                                             |
        -------X--------------.------corr----X----------X--------------.-------
    TARGET     |              |       ||                |              |
       ta1 |0> .--MX  ta1 |+> X--MZ ===         ta3 |0> .--MX  ta4 |+> X--MZ
    '''
    global dir_error_rate, factor, bs
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_before_logical_errors = 0
    total_num_errors_control = 0; total_num_errors_target = 0
    num_shots = num_batch * bs
    import pickle
    with open(f"logs_prep_SPAM_equal_CNOT/d15_zero/propagation_dict.pkl", 'rb') as f:
        d15_zero_prop_dict = pickle.load(f)
    with open(f"logs_prep_SPAM_equal_CNOT/d15_plus/propagation_dict.pkl", 'rb') as f:
        d15_plus_prop_dict = pickle.load(f)
    start = time.time()
    ca1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index, dir_error_rate, factor, postprocess=False)
    ca2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index, dir_error_rate, factor, postprocess=False)
    ca3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index+1, dir_error_rate, factor, postprocess=False)
    ca4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index+1, dir_error_rate, factor, postprocess=False)
    ta1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index+2, dir_error_rate, factor, postprocess=False)
    ta2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index+2, dir_error_rate, factor, postprocess=False)
    ta3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 4*index+3, dir_error_rate, factor, postprocess=False)
    ta4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 4*index+3, dir_error_rate, factor, postprocess=False)
    avg_c_Z_flip, avg_t_Z_flip = [], []
    avg_c_X_flip, avg_t_X_flip = [], []
    num_ZI, num_IZ, num_ZZ, num_IX, num_XI, num_XX = 0,0,0,0,0,0
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1_control, error_mask_before_log_phase_control = phase_flip_EC_block(ancilla_errors=process_ancilla_error(d15_zero_prop_dict, ca1_d15_zero[s_start:s_end]), decoder=decoder_r3, alert=True) # TODO: expect same LER when turn disable_correction_error=True
        residual_errors_b2_control, error_mask_before_log_bit_control = bit_flip_EC_block(input_errors=residual_errors_b1_control, ancilla_errors=process_ancilla_error(d15_plus_prop_dict, ca2_d15_plus[s_start:s_end]), disable_correction_error=True, decoder=decoder_r3, alert=True)
        residual_errors_b1_target, error_mask_before_log_phase_target = phase_flip_EC_block(ancilla_errors=process_ancilla_error(d15_zero_prop_dict, ta1_d15_zero[s_start:s_end]), decoder=decoder_r3, alert=True) # TODO: disable_correction_error=True
        residual_errors_b2_target, error_mask_before_log_bit_target = bit_flip_EC_block(input_errors=residual_errors_b1_target, ancilla_errors=process_ancilla_error(d15_plus_prop_dict, ta2_d15_plus[s_start:s_end]), disable_correction_error=True, decoder=decoder_r3, alert=True)
        residual_errors_control, residual_errors_target = logical_CNOT(residual_errors_b2_control, residual_errors_b2_target)
        residual_errors_b3_control, error_mask_phase_control = phase_flip_EC_block(input_errors=residual_errors_control, ancilla_errors=process_ancilla_error(d15_zero_prop_dict, ca3_d15_zero[s_start:s_end]), decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_c_Z_flip) # preceding block is CNOT, not LEC
        residual_errors_b4_control, error_mask_bit_control = bit_flip_EC_block(input_errors=residual_errors_b3_control, ancilla_errors=process_ancilla_error(d15_plus_prop_dict, ca4_d15_plus[s_start:s_end]), decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_c_X_flip)
        residual_errors_b3_target, error_mask_phase_target = phase_flip_EC_block(input_errors=residual_errors_target, ancilla_errors=process_ancilla_error(d15_zero_prop_dict, ta3_d15_zero[s_start:s_end]), decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_t_Z_flip)
        residual_errors_b4_target, error_mask_bit_target = bit_flip_EC_block(input_errors=residual_errors_b3_target, ancilla_errors=process_ancilla_error(d15_plus_prop_dict, ta4_d15_plus[s_start:s_end]), decoder=decoder_r3, disable_correction_error=True, avg_flip=avg_t_X_flip)
        control_errors = np.logical_or(error_mask_phase_control, error_mask_bit_control)
        target_errors = np.logical_or(error_mask_phase_target, error_mask_bit_target)
        total_after_logical_error = np.logical_or(control_errors, target_errors)
        total_num_errors += total_after_logical_error.astype(int).sum()
        total_before_logical_error = np.logical_or(np.logical_or(error_mask_before_log_phase_control, error_mask_before_log_bit_control), np.logical_or(error_mask_before_log_phase_target, error_mask_before_log_bit_target))
        total_before_logical_errors += total_before_logical_error.astype(int).sum()
        total_num_errors_control += sum(control_errors)
        total_num_errors_target += sum(target_errors)
        num_ZI += sum(error_mask_phase_control)
        num_XI += sum(error_mask_bit_control)
        num_IZ += sum(error_mask_phase_target)
        num_IX += sum(error_mask_bit_target)
        num_ZZ += sum(np.logical_and(error_mask_phase_control, error_mask_phase_target))
        num_XX += sum(np.logical_and(error_mask_bit_control, error_mask_bit_target))
        if total_before_logical_error.any():
            print(f"before logical error happened on sample {np.where(total_before_logical_error)[0]} within batch")
            if error_mask_before_log_phase_control.any():
                print("on control phase EC block, ancilla error", process_ancilla_error(d15_zero_prop_dict, [ca1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_control)[0]], decoder=decoder_r3))
            if error_mask_before_log_phase_target.any():
                print("on target phase EC block, ancilla error", process_ancilla_error(d15_zero_prop_dict, [ta1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_target)[0]], decoder=decoder_r3))
            if error_mask_before_log_bit_control.any():
                print("on control bit EC block, ancilla 1 error", process_ancilla_error(d15_zero_prop_dict, [ca1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_control)[0]], decoder=decoder_r3))
                print("ancilla 2 error", process_ancilla_error(d15_plus_prop_dict, [ca2_d15_plus[s_start+i] for i in np.where(error_mask_before_log_bit_control)[0]], decoder=decoder_r3))
            if error_mask_before_log_bit_target.any():
                print("on target bit EC block, ancilla 1 error", process_ancilla_error(d15_zero_prop_dict, [ta1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_target)[0]], decoder=decoder_r3))
                print("ancilla 2 error", process_ancilla_error(d15_plus_prop_dict, [ta2_d15_plus[s_start+i] for i in np.where(error_mask_before_log_bit_target)[0]], decoder=decoder_r3))
            print(f"after logical error happened on sample {np.where(total_after_logical_error)[0]} within batch", flush=True)
    end = time.time()
    print(f"#errors before logical/#samples: {total_before_logical_errors}/{s_end}")
    print(f"#errors/#samples: {total_num_errors}/{s_end}")
    # print(f"#control errors/#samples: {total_num_errors_control}/{s_end}, #phase flip: {sum(error_mask_phase_control)}, #bit flip: {sum(error_mask_bit_control)}")
    # print(f"#target errors/#samples: {total_num_errors_target}/{s_end}, #phase flip: {sum(error_mask_phase_target)}, #bit flip: {sum(error_mask_bit_target)}")
    print(f"ZI: {num_ZI}, XI: {num_XI}, IZ: {num_IZ}, IX: {num_IX}, ZZ: {num_ZZ}, XX: {num_XX}")
    print(f"error rate: {total_num_errors/(s_end)}")
    print(f"control average Z flips: {sum(avg_c_Z_flip)/(round+1)}, X flips: {sum(avg_c_X_flip)/(round+1)}, target average Z flips: {sum(avg_t_Z_flip)/(round+1)}, X flips: {sum(avg_t_X_flip)/(round+1)}")
    print(f"Total elasped time {end-start} seconds.", flush=True)

def simulate_CNOT_rectangle_bit_first(num_batch=200): # extended rectangle for logical transversal CNOT between two blocks
    # correct bit-flip first, then phase-flip
    # control phase-flip > target bit-flip. ZI > IX
    ''' -------.--------------X------corr----.----------.--------------X-------
    CONTROL    |              |       ||     |          |              |
       ca1 |+> X--MZ  ca2 |0> .--MX ===      |  ca3 |+> X--MZ  ca4 |0> .--MX
                                             |
        -------.--------------X------corr----X----------.--------------X-------
    TARGET     |              |       ||                |              |
       ta1 |+> X--MZ  ta1 |0> .--MX ===         ta3 |+> X--MZ  ta4 |0> .--MX
    '''
    global dir_error_rate, factor
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_num_errors_control = 0; total_num_errors_target = 0
    num_shots = num_batch * bs
    start = time.time()
    ca1_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 0, dir_error_rate, factor)
    ca2_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 0, dir_error_rate, factor)
    ca3_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 1, dir_error_rate, factor)
    ca4_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 1, dir_error_rate, factor)
    ta1_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 2, dir_error_rate, factor)
    ta2_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 2, dir_error_rate, factor)
    ta3_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 3, dir_error_rate, factor)
    ta4_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 3, dir_error_rate, factor)
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1_control = bit_flip_EC_block(ancilla_errors=ca1_d15_plus[s_start:s_end])
        residual_errors_b2_control = phase_flip_EC_block(input_errors=residual_errors_b1_control, ancilla_errors=ca2_d15_zero[s_start:s_end], disable_correction_error=True)
        residual_errors_b1_target = bit_flip_EC_block(ancilla_errors=ta1_d15_plus[s_start:s_end])
        residual_errors_b2_target = phase_flip_EC_block(input_errors=residual_errors_b1_target, ancilla_errors=ta2_d15_zero[s_start:s_end], disable_correction_error=True)
        residual_errors_control, residual_errors_target = logical_CNOT(residual_errors_b2_control, residual_errors_b2_target)
        residual_errors_b3_control, error_mask_bit_control = bit_flip_EC_block(input_errors=residual_errors_control, ancilla_errors=ca3_d15_plus[s_start:s_end], decoder=decoder_r3, disable_correction_error=True)
        residual_errors_b4_control, error_mask_phase_control = phase_flip_EC_block(input_errors=residual_errors_b3_control, ancilla_errors=ca4_d15_zero[s_start:s_end], decoder=decoder_r3, disable_correction_error=True)
        residual_errors_b3_target, error_mask_bit_target = bit_flip_EC_block(input_errors=residual_errors_target, ancilla_errors=ta3_d15_plus[s_start:s_end], decoder=decoder_r3, disable_correction_error=True)
        residual_errors_b4_target, error_mask_phase_target = phase_flip_EC_block(input_errors=residual_errors_b3_target, ancilla_errors=ta4_d15_zero[s_start:s_end], decoder=decoder_r3, disable_correction_error=True)
        control_errors = np.logical_or(error_mask_phase_control, error_mask_bit_control)
        target_errors = np.logical_or(error_mask_phase_target, error_mask_bit_target)
        total_num_errors += np.logical_or(control_errors, target_errors).astype(int).sum()
        total_num_errors_control += sum(control_errors)
        total_num_errors_target += sum(target_errors)
    end = time.time()
    print(f"#errors/#samples: {total_num_errors}/{s_end}")
    print(f"#control errors/#samples: {total_num_errors_control}/{s_end}, #phase flip: {sum(error_mask_phase_control)}, #bit flip: {sum(error_mask_bit_control)}")
    print(f"#target errors/#samples: {total_num_errors_target}/{s_end}, #phase flip: {sum(error_mask_phase_target)}, #bit flip: {sum(error_mask_bit_target)}")
    print(f"error rate: {total_num_errors/(s_end)}")
    print(f"Total elasped time {end-start} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Simulate extended rectangle (exRec) for Clifford and T logical gates")
    parser.add_argument("--factor", type=float, choices=[1.0, 0.5], default=1.0, help="the ratio of p_SPAM to p_CNOT, use 1.0 or 0.5")
    parser.add_argument("-fs", "--factor_single", type=float, choices=[0.1, 0.2, 1.0], default=1.0, help="the ratio of p_single to p_CNOT, choose between [0.1, 0.2, 1.0]")
    parser.add_argument("-fc", "--factor_correction", type=float, choices=[0.0, 0.1, 0.2, 1.0], default=1.0, help="the ratio of p_correction to p_CNOT, choose between [0.0, 0.1, 0.2, 1.0]")
    parser.add_argument("-bs", "--batch_size", type=int, default=1024, help="batch size, please use a multiple of 256, default to 1024")
    parser.add_argument("--index", type=int, default=0, help="index of the file")
    parser.add_argument("--num_batch", type=int, default=100, help="number of batch")
    parser.add_argument("--p_CNOT", type=float, help="physical error rate of CNOT")
    parser.add_argument("-t", "--rec_type", choices=["CNOT", "H", "S", "T"], help="type of the exRec, choose between [CNOT, H, S, T]")
    parser.add_argument("-pft", "--pauli_frame_tracking", choices=[True, False], default=False, help="whether turn on Pauli Frame Tracking for Clifford gates, default to True")
    args = parser.parse_args()

    num_batch = args.num_batch
    factor = args.factor
    factor_single = args.factor_single
    factor_correction = args.factor_correction
    p_CNOT = args.p_CNOT 
    p_SPAM = factor * p_CNOT
    p_single = factor_single * p_CNOT # single qubit gate, H, S, T, transversal on all qubits
    p_correction = factor_correction * p_CNOT # single qubit addressing, arbitrary Pauli string
    rec_type = args.rec_type
    pft = args.pauli_frame_tracking
    print(f"p_CNOT={p_CNOT}, p_SPAM={p_SPAM}, p_single={p_single}, p_correction={p_correction}, exRec type={rec_type}, Pauli Frame Tracking={pft}, index={args.index}")
    dir_error_rate = "p" + str(p_CNOT).split('.')[1]
    if rec_type in ["H", "S"]:
        simulate_single_qubit_Clifford_rectangle(gate_type=rec_type)

    elif rec_type == "CNOT":
        print("running CNOT exRec, index", args.index)
        simulate_CNOT_rectangle_special_load(num_batch=num_batch, index=args.index)
        # for s in range(25):
        #     simulate_CNOT_rectangle(num_batch=num_batch, index=s)
        # simulate_CNOT_rectangle_bit_first()

    else: # T gate implemented through code switching
        # simulate_code_switching_rectangle(num_batch=num_batch, index=args.index)
        print("running code switching exRec implemented with special load")
        simulate_code_switching_rectangle_special_load(num_batch=num_batch, index=args.index) # for p=0.0002

    # for s in range(20):
        # simulate_code_switching_rectangle(num_batch=1, index=index+s)
        # simulate_CNOT_rectangle(num_batch=1, index=index+s)