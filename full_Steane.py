import stim
print(stim.__version__)
import numpy as np
import time, sys, argparse
from PyDecoder_polar import PyDecoder_polar_SCL
from utils import z_component, x_component, AncillaErrorLoader

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
    global bs
    x_wt = [len(s.pauli_indices('XY')) for s in errors]
    x = [s.pauli_indices('XY') for s in errors]
    z_wt = [len(s.pauli_indices('ZY')) for s in errors]
    z = [s.pauli_indices('YZ') for s in errors]
    y_wt = [len(s.pauli_indices('Y')) for s in errors]
    y = [s.pauli_indices('Y') for s in errors]
    print(f"mean XY wt: {sum(x_wt)/bs}, mean YZ wt: {sum(z_wt)/bs}, mean pure Y component: {sum(y_wt)/bs}")
    # print(f"XY component {x}, YZ component {z}, pure Y component {y}")

# phase error Steane EC block
def phase_flip_EC_block(ancilla_errors, input_errors=None, decoder=None, enable_correction_error=False, avg_flip=None, alert=False):
    global p_CNOT, p_SPAM, p_correction, bs
    if input_errors is None:
        input_errors = [stim.PauliString(N) for _ in range(bs)]
    # broadcast incoming noise to qubit 0 ~ N-1
    # broadcast ancilla noise to qubit N ~ 2N-1
    # print("phase EC block input noise")
    # mean_wt(input_errors)
    before_coupling_errors = [e1+e2 for (e1,e2) in zip(input_errors, ancilla_errors)] # list of length 2N PauliString
    
    coupling_circuit = stim.Circuit()
    if enable_correction_error: 
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

def bit_flip_EC_block(ancilla_errors, input_errors=None, decoder=None, enable_correction_error=False, avg_flip=[], alert=False):
    global p_CNOT, p_SPAM, p_correction, bs
    if input_errors is None:
        input_errors = [stim.PauliString(N) for _ in range(bs)]
    # broadcast incoming noise to qubit 0 ~ N-1
    # broadcast ancilla noise to qubit N ~ 2N-1
    # print("bit EC block input noise")
    # mean_wt(input_errors)
    before_coupling_errors = [e1+e2 for (e1,e2) in zip(input_errors, ancilla_errors)] # list of length 2N PauliString
    
    coupling_circuit = stim.Circuit()
    if enable_correction_error: 
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
    global p_CNOT, p_SPAM, p_single, p_correction, bs
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
    global p_CNOT, p_SPAM, p_correction, bs
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
    global p_CNOT, p_SPAM, p_single, p_correction, bs
    single_qubit_correction_noise_circuit = stim.Circuit()
    for i in range(N-1): # previous round correction noise (for code switching)
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
    global dir_error_rate, factor, bs
    decoder_r4 = PyDecoder_polar_SCL(4)
    decoder_r3 = PyDecoder_polar_SCL(3)
    decoder_r2 = PyDecoder_polar_SCL(2)
    parent_dir = "logs_prep_SPAM_equal_CNOT" if factor == 1.0 else "logs_prep_SPAM_half_CNOT"
    d15_zero_parent_dir = parent_dir + f"/d15_zero/{dir_error_rate}"
    d15_plus_parent_dir = parent_dir + f"/d15_plus/{dir_error_rate}"
    d7_plus_parent_dir = parent_dir + f"/d7_plus/{dir_error_rate}"
    loader = AncillaErrorLoader(decoder_r3, decoder_r2=decoder_r2, decoder_r4=decoder_r4)
    total_num_errors = 0
    total_Z_errors = 0; total_X_errors = 0 # want to confirm the phase flip is the dominant term
    num_shots = num_batch * bs
    start = time.time()
    a1_d15_zero = loader.sample_ancilla_error(num_shots, 2*index, d15_zero_parent_dir)
    a2_d7_plus = loader.sample_ancilla_error(num_shots, index, d7_plus_parent_dir)
    a3_d15_zero = loader.sample_ancilla_error(num_shots, 2*index+1, d15_zero_parent_dir)
    a4_d15_plus = loader.sample_ancilla_error(num_shots, index, d15_plus_parent_dir)
    avg_Z_flip = []
    avg_X_flip = []
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1 = phase_flip_EC_block(ancilla_errors=loader.process_ancilla_error(a1_d15_zero[s_start:s_end], 15, 'zero'))
        residual_errors_b2 = bit_flip_EC_block(input_errors=residual_errors_b1, ancilla_errors=loader.process_ancilla_error(a2_d7_plus[s_start:s_end], 7, 'plus'))
        residual_errors = logical_T(residual_errors_b2)
        residual_errors_b3, error_mask_phase = phase_flip_EC_block(input_errors=residual_errors, ancilla_errors=loader.process_ancilla_error(a3_d15_zero[s_start:s_end], 15, 'zero'), decoder=decoder_r4, avg_flip=avg_Z_flip)
        residual_errors_b4, error_mask_bit = bit_flip_EC_block(input_errors=residual_errors_b3, ancilla_errors=loader.process_ancilla_error(a4_d15_plus[s_start:s_end], 15, 'plus'), enable_correction_error=True, decoder=decoder_r3, avg_flip=avg_X_flip)
        total_num_errors += np.logical_or(error_mask_phase, error_mask_bit).astype(int).sum()
        total_Z_errors += sum(error_mask_phase)
        total_X_errors += sum(error_mask_bit)
    end = time.time()
    print(f"#errors/#samples: {total_num_errors}/{s_end}")
    print(f"#phase errors/#samples: {total_Z_errors}/{s_end}, #bit errors/#samples: {total_X_errors}/{s_end}")
    print(f"error rate: {total_num_errors/(s_end)}")
    print(f"average Z flips: {sum(avg_Z_flip)/(round+1)}, average X flips: {sum(avg_X_flip)/(round+1)}")
    print(f"Total elasped time {end-start} seconds.")


def simulate_single_qubit_Clifford_rectangle(num_batch=150, gate_type='H', index=0): # extended rectangle for logical transversal H or S
    if gate_type not in ['H', 'S']:
        print("Unsupported gate type")
        return
    global dir_error_rate, factor, bs
    decoder_r3 = PyDecoder_polar_SCL(3)
    parent_dir = "logs_prep_SPAM_equal_CNOT" if factor == 1.0 else "logs_prep_SPAM_half_CNOT"
    d15_zero_parent_dir = parent_dir + f"/d15_zero/{dir_error_rate}"
    d15_plus_parent_dir = parent_dir + f"/d15_plus/{dir_error_rate}"
    loader = AncillaErrorLoader(decoder_r3)
    total_num_errors = 0
    total_Z_errors = 0; total_X_errors = 0 # want to confirm the phase flip is the dominant term
    total_before_logical_errors = 0
    num_shots = num_batch * bs
    start = time.time()
    a1_d15_zero = loader.sample_ancilla_error(num_shots, 2*index, d15_zero_parent_dir)
    a2_d15_plus = loader.sample_ancilla_error(num_shots, 2*index, d15_plus_parent_dir)
    a3_d15_zero = loader.sample_ancilla_error(num_shots, 2*index+1, d15_zero_parent_dir)
    a4_d15_plus = loader.sample_ancilla_error(num_shots, 2*index+1, d15_plus_parent_dir)
    avg_Z_flip = []
    avg_X_flip = []
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1, error_mask_before_log_phase = phase_flip_EC_block(ancilla_errors=loader.process_ancilla_error(a1_d15_zero[s_start:s_end], 15, 'zero'), decoder=decoder_r3, alert=True)
        residual_errors_b2, error_mask_before_log_bit = bit_flip_EC_block(input_errors=residual_errors_b1, ancilla_errors=loader.process_ancilla_error(a2_d15_plus[s_start:s_end], 15, 'plus'), decoder=decoder_r3, alert=True)
        residual_errors = logical_single_qubit_Clifford(residual_errors_b2, gate_type)
        residual_errors_b3, error_mask_phase = phase_flip_EC_block(input_errors=residual_errors, ancilla_errors=loader.process_ancilla_error(a3_d15_zero[s_start:s_end], 15, 'zero'), decoder=decoder_r3, avg_flip=avg_Z_flip)
        residual_errors_b4, error_mask_bit = bit_flip_EC_block(input_errors=residual_errors_b3, ancilla_errors=loader.process_ancilla_error(a4_d15_plus[s_start:s_end], 15, 'plus'), decoder=decoder_r3, avg_flip=avg_X_flip)
        total_num_errors += np.logical_or(error_mask_phase, error_mask_bit).astype(int).sum()
        total_Z_errors += sum(error_mask_phase)
        total_X_errors += sum(error_mask_bit)
        total_before_logical_error = np.logical_or(error_mask_before_log_phase, error_mask_before_log_bit)
        total_before_logical_errors += total_before_logical_error.astype(int).sum()
    end = time.time()
    print(f"#errors before logical/#samples: {total_before_logical_errors}/{s_end}")
    print(f"#errors/#samples: {total_num_errors}/{s_end}")
    print(f"#phase errors/#samples: {total_Z_errors}/{s_end}, #bit errors/#samples: {total_X_errors}/{s_end}")
    print(f"error rate: {total_num_errors/(s_end)}")
    print(f"average Z flips: {sum(avg_Z_flip)/(round+1)}, average X flips: {sum(avg_X_flip)/(round+1)}")
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
    parent_dir = "logs_prep_SPAM_equal_CNOT" if factor == 1.0 else "logs_prep_SPAM_half_CNOT"
    d15_zero_parent_dir = parent_dir + f"/d15_zero/{dir_error_rate}"
    d15_plus_parent_dir = parent_dir + f"/d15_plus/{dir_error_rate}"
    loader = AncillaErrorLoader(decoder_r3)
    total_num_errors = 0
    total_before_logical_errors = 0
    num_shots = num_batch * bs
    start = time.time()
    ca1_d15_zero = loader.sample_ancilla_error(num_shots, 4*index, d15_zero_parent_dir)
    ca2_d15_plus = loader.sample_ancilla_error(num_shots, 4*index, d15_plus_parent_dir)
    ca3_d15_zero = loader.sample_ancilla_error(num_shots, 4*index+1, d15_zero_parent_dir)
    ca4_d15_plus = loader.sample_ancilla_error(num_shots, 4*index+1, d15_plus_parent_dir)
    ta1_d15_zero = loader.sample_ancilla_error(num_shots, 4*index+2, d15_zero_parent_dir)
    ta2_d15_plus = loader.sample_ancilla_error(num_shots, 4*index+2, d15_plus_parent_dir)
    ta3_d15_zero = loader.sample_ancilla_error(num_shots, 4*index+3, d15_zero_parent_dir)
    ta4_d15_plus = loader.sample_ancilla_error(num_shots, 4*index+3, d15_plus_parent_dir)
    avg_c_Z_flip, avg_t_Z_flip = [], []
    avg_c_X_flip, avg_t_X_flip = [], []
    num_ZI, num_IZ, num_ZZ, num_IX, num_XI, num_XX = 0,0,0,0,0,0
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1_control, error_mask_before_log_phase_control = phase_flip_EC_block(ancilla_errors=loader.process_ancilla_error(ca1_d15_zero[s_start:s_end], 15, 'zero'), decoder=decoder_r3, alert=True) 
        residual_errors_b2_control, error_mask_before_log_bit_control = bit_flip_EC_block(input_errors=residual_errors_b1_control, ancilla_errors=loader.process_ancilla_error(ca2_d15_plus[s_start:s_end], 15, 'plus'), decoder=decoder_r3, alert=True)
        residual_errors_b1_target, error_mask_before_log_phase_target = phase_flip_EC_block(ancilla_errors=loader.process_ancilla_error(ta1_d15_zero[s_start:s_end], 15, 'zero'), decoder=decoder_r3, alert=True)
        residual_errors_b2_target, error_mask_before_log_bit_target = bit_flip_EC_block(input_errors=residual_errors_b1_target, ancilla_errors=loader.process_ancilla_error(ta2_d15_plus[s_start:s_end], 15, 'plus'), decoder=decoder_r3, alert=True)
        residual_errors_control, residual_errors_target = logical_CNOT(residual_errors_b2_control, residual_errors_b2_target)
        residual_errors_b3_control, error_mask_phase_control = phase_flip_EC_block(input_errors=residual_errors_control, ancilla_errors=loader.process_ancilla_error(ca3_d15_zero[s_start:s_end], 15, 'zero'), decoder=decoder_r3, avg_flip=avg_c_Z_flip) # preceding block is CNOT, not LEC
        residual_errors_b4_control, error_mask_bit_control = bit_flip_EC_block(input_errors=residual_errors_b3_control, ancilla_errors=loader.process_ancilla_error(ca4_d15_plus[s_start:s_end], 15, 'plus'), decoder=decoder_r3, avg_flip=avg_c_X_flip)
        residual_errors_b3_target, error_mask_phase_target = phase_flip_EC_block(input_errors=residual_errors_target, ancilla_errors=loader.process_ancilla_error(ta3_d15_zero[s_start:s_end], 15, 'zero'), decoder=decoder_r3, avg_flip=avg_t_Z_flip)
        residual_errors_b4_target, error_mask_bit_target = bit_flip_EC_block(input_errors=residual_errors_b3_target, ancilla_errors=loader.process_ancilla_error(ta4_d15_plus[s_start:s_end], 15, 'plus'), decoder=decoder_r3, avg_flip=avg_t_X_flip)
        control_errors = np.logical_or(error_mask_phase_control, error_mask_bit_control)
        target_errors = np.logical_or(error_mask_phase_target, error_mask_bit_target)
        total_after_logical_error = np.logical_or(control_errors, target_errors)
        total_num_errors += total_after_logical_error.astype(int).sum()
        total_before_logical_error = np.logical_or(np.logical_or(error_mask_before_log_phase_control, error_mask_before_log_bit_control), np.logical_or(error_mask_before_log_phase_target, error_mask_before_log_bit_target))
        total_before_logical_errors += total_before_logical_error.astype(int).sum()
        num_ZI += sum(error_mask_phase_control)
        num_XI += sum(error_mask_bit_control)
        num_IZ += sum(error_mask_phase_target)
        num_IX += sum(error_mask_bit_target)
        num_ZZ += sum(np.logical_and(error_mask_phase_control, error_mask_phase_target))
        num_XX += sum(np.logical_and(error_mask_bit_control, error_mask_bit_target))
        if total_before_logical_error.any():
            print(f"before logical error happened on sample {np.where(total_before_logical_error)[0]} within batch")
            if error_mask_before_log_phase_control.any():
                print("on control phase EC block, ancilla error", loader.process_ancilla_error([ca1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_control)[0]], 15, 'zero'))
            if error_mask_before_log_phase_target.any():
                print("on target phase EC block, ancilla error", loader.process_ancilla_error([ta1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_target)[0]], 15, 'zero'))
            if error_mask_before_log_bit_control.any():
                print("on control bit EC block, ancilla 1 error", loader.process_ancilla_error([ca1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_control)[0]], 15, 'zero'))
                print("ancilla 2 error", loader.process_ancilla_error([ca2_d15_plus[s_start+i] for i in np.where(error_mask_before_log_bit_control)[0]], 15, 'plus'))
            if error_mask_before_log_bit_target.any():
                print("on target bit EC block, ancilla 1 error", loader.process_ancilla_error([ta1_d15_zero[s_start+i] for i in np.where(error_mask_before_log_phase_target)[0]], 15, 'zero'))
                print("ancilla 2 error", loader.process_ancilla_error([ta2_d15_plus[s_start+i] for i in np.where(error_mask_before_log_bit_target)[0]], 15, 'plus'))
            print(f"after logical error happened on sample {np.where(total_after_logical_error)[0]} within batch", flush=True)
    end = time.time()
    print(f"#errors before logical/#samples: {total_before_logical_errors}/{s_end}")
    print(f"#errors/#samples: {total_num_errors}/{s_end}")
    print(f"ZI: {num_ZI}, XI: {num_XI}, IZ: {num_IZ}, IX: {num_IX}, ZZ: {num_ZZ}, XX: {num_XX}")
    print(f"error rate: {total_num_errors/(s_end)}")
    print(f"control average Z flips: {sum(avg_c_Z_flip)/(round+1)}, X flips: {sum(avg_c_X_flip)/(round+1)}, target average Z flips: {sum(avg_t_Z_flip)/(round+1)}, X flips: {sum(avg_t_X_flip)/(round+1)}")
    print(f"Total elasped time {end-start} seconds.", flush=True)


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
    args = parser.parse_args()

    num_batch = args.num_batch
    factor = args.factor
    bs = args.batch_size
    factor_single = args.factor_single
    factor_correction = args.factor_correction
    p_CNOT = args.p_CNOT 
    p_SPAM = factor * p_CNOT
    p_single = factor_single * p_CNOT # single qubit gate, H, S, T, transversal on all qubits
    p_correction = factor_correction * p_CNOT # single qubit addressing, arbitrary Pauli string
    rec_type = args.rec_type
    print(f"p_CNOT={p_CNOT}, p_SPAM={p_SPAM}, p_single={p_single}, p_correction={p_correction}, exRec type={rec_type}, index={args.index}")
    dir_error_rate = "p" + str(p_CNOT).split('.')[1]
    if rec_type in ["H", "S"]:
        print(f"running {rec_type} exRec, index", args.index)
        simulate_single_qubit_Clifford_rectangle(num_batch=num_batch, index=args.index, gate_type=rec_type)

    elif rec_type == "CNOT":
        print("running CNOT exRec, index", args.index)
        simulate_CNOT_rectangle(num_batch=num_batch, index=args.index)

    else: # T gate implemented through code switching
        print("running code switching exRec, index", args.index)
        simulate_code_switching_rectangle(num_batch=num_batch, index=args.index)

