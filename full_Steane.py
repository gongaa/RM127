import stim
print(stim.__version__)
import numpy as np
import time
from PyDecoder_polar import PyDecoder_polar_SCL
from utils import z_component, x_component, sample_ancilla_error

N = 2 ** 7
p_CNOT = 0.005
p_single = p_CNOT/2


bs = 1024
# phase error Steane EC block
def phase_flip_EC_block(ancilla_errors, input_errors=None, decoder=None, disable_correction_error=False):
    if input_errors is None:
        input_errors = [stim.PauliString(N) for _ in range(bs)]
    # broadcast incoming noise to qubit 0 ~ N-1
    # broadcast ancilla noise to qubit N ~ 2N-1
    before_coupling_errors = [e1+e2 for (e1,e2) in zip(input_errors, ancilla_errors)] # list of length 2N PauliString
    
    coupling_circuit = stim.Circuit()
    if not disable_correction_error: 
        # for example: first bit-flip then phase-flip EC, 
        # then correction can be applied when both blocks are done, 
        # sparing the correction error for the second block
        for i in range(N-1): # when applying previous round error correction operator, there is single-qubit noise
            coupling_circuit.append("DEPOLARIZE1", i, p_single)
    for i in range(N-1):
        coupling_circuit.append("CNOT", [i+N, i])
        coupling_circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)

    for i in range(N-1): # measurement noise
        coupling_circuit.append("Z_ERROR", N+i, p_single)
    coupling_sim = stim.FlipSimulator(batch_size=bs, num_qubits=2*N, disable_stabilizer_randomization=True)
    
    X_component, Z_component = np.array([e.to_numpy() for e in before_coupling_errors]).transpose(1,2,0) # each shape (2*N, bs)
    coupling_sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    coupling_sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    coupling_sim.do(coupling_circuit)
    coupling_errors = coupling_sim.peek_pauli_flips()

    if decoder is not None:
        # look at Z component on qubit N ~ 2N-1 and try to correct
        noise = list(map(lambda s: z_component(s[N:]), coupling_errors))
        error_mask = [False for _ in range(bs)]
        for i in range(len(noise)):
            num_flip = decoder.decode(list(np.nonzero(noise[i])[0]))
            class_bit = decoder.last_info_bit
            if class_bit != 0:
                error_mask[i] = True
                print("# phase flip:", num_flip)
        residual_errors = list(map(lambda s: s[:N] * z_component(s[N:]), coupling_errors))
        return residual_errors, error_mask
    else: # assume perfect correction first
        residual_errors = list(map(lambda s: s[:N] * z_component(s[N:]), coupling_errors))
        return residual_errors

def bit_flip_EC_block(ancilla_errors, input_errors=None, decoder=None, disable_correction_error=False):
    if input_errors is None:
        input_errors = [stim.PauliString(N) for _ in range(bs)]
    # broadcast incoming noise to qubit 0 ~ N-1
    # broadcast ancilla noise to qubit N ~ 2N-1
    before_coupling_errors = [e1+e2 for (e1,e2) in zip(input_errors, ancilla_errors)] # list of length 2N PauliString
    
    coupling_circuit = stim.Circuit()
    if not disable_correction_error: 
        for i in range(N-1): # previous round EC operation noise
            coupling_circuit.append("DEPOLARIZE1", i, p_single)
    for i in range(N-1):
        coupling_circuit.append("CNOT", [i, i+N])
        coupling_circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)

    for i in range(N-1): # measurement noise
        coupling_circuit.append("X_ERROR", N+i, p_single)
    coupling_sim = stim.FlipSimulator(batch_size=bs, num_qubits=2*N, disable_stabilizer_randomization=True)
    
    X_component, Z_component = np.array([e.to_numpy() for e in before_coupling_errors]).transpose(1,2,0) # each shape (2*N, bs)
    coupling_sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    coupling_sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    coupling_sim.do(coupling_circuit)
    coupling_errors = coupling_sim.peek_pauli_flips()

    if decoder is not None:
        # look at X component on qubit N ~ 2N-1 and try to correct
        noise = list(map(lambda s: x_component(s[N:]), coupling_errors))
        error_mask = [False for _ in range(bs)]
        for i in range(len(noise)):
            num_flip = decoder.decode(list(np.nonzero(noise[i])[0]))
            class_bit = decoder.last_info_bit
            if class_bit != 0:
                error_mask[i] = True
                print("# bit flip:", num_flip)
        residual_errors = list(map(lambda s: s[:N] * x_component(s[N:]), coupling_errors))
        return residual_errors, error_mask
    else: # assume perfect correction first
        residual_errors = list(map(lambda s: s[:N] * x_component(s[N:]), coupling_errors))
        
    return residual_errors


def logical_single_qubit_Clifford(input_noise, gate_type='H'): # transversal Hadamard or S gate, without leading/trailing EC 
    if gate_type not in ['H', 'S']:
        print("Unsupported gate type")
        return
    transversal_logical_circuit = stim.Circuit()
    for i in range(N-1): # previous round EC operation noise
        transversal_logical_circuit.append("DEPOLARIZE1", i, p_single)
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
    transversal_CNOT_circuit = stim.Circuit()
    for i in range(N-1): # previous round EC operation noise
        transversal_CNOT_circuit.append("DEPOLARIZE1", i, p_single)
        transversal_CNOT_circuit.append("DEPOLARIZE1", i+N, p_single)
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
    
    
def simulate_code_switching_rectangle(num_batch=200):    # batch size fixed to 1024
    decoder_r4 = PyDecoder_polar_SCL(4)
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_Z_errors = 0; total_X_errors = 0 # want to confirm the phase flip is the dominant term
    num_shots = num_batch * bs
    start = time.time()
    a1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 0)
    a2_d7_plus = sample_ancilla_error(num_shots, 7, 'plus', 0)
    a3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 1)
    a4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 0)
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1 = phase_flip_EC_block(ancilla_errors=a1_d15_zero[s_start:s_end])
        residual_errors_b2 = bit_flip_EC_block(residual_errors_b1, a2_d7_plus[s_start:s_end], disable_correction_error=True)
        # TODO: logical T gate, currently using S
        residual_errors = logical_single_qubit_Clifford(residual_errors_b2, 'S')
        residual_errors_b3, error_mask_phase = phase_flip_EC_block(residual_errors, a3_d15_zero[s_start:s_end], decoder=decoder_r4)
        residual_errors_b4, error_mask_bit = bit_flip_EC_block(residual_errors_b3, a4_d15_plus[s_start:s_end], decoder=decoder_r3, disable_correction_error=True)
        total_num_errors += np.logical_or(error_mask_phase, error_mask_bit).astype(int).sum()
        total_Z_errors += sum(error_mask_phase)
        total_X_errors += sum(error_mask_bit)
        print(f"#errors/#samples: {total_num_errors}/{s_end}")
        print(f"#phase errors/#samples: {total_Z_errors}/{s_end}, #bit errors/#samples: {total_X_errors}/{s_end}")
        print(f"error rate: {total_num_errors/(s_end)}")
    end = time.time()
    print(f"Total elasped time {end-start} seconds.")



def simulate_single_qubit_Clifford_rectangle(num_batch=200, gate_type='H'): # extended rectangle for logical transversal H or S
    if gate_type not in ['H', 'S']:
        print("Unsupported gate type")
        return
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_Z_errors = 0; total_X_errors = 0 # want to confirm the phase flip is the dominant term
    num_shots = num_batch * bs
    start = time.time()
    a1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 0)
    a2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 0)
    a3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 1)
    a4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 1)
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1 = phase_flip_EC_block(ancilla_errors=a1_d15_zero[s_start:s_end])
        residual_errors_b2 = bit_flip_EC_block(residual_errors_b1, a2_d15_plus[s_start:s_end])
        residual_errors = logical_single_qubit_Clifford(residual_errors_b2, gate_type)
        residual_errors_b3, error_mask_phase = phase_flip_EC_block(residual_errors, a3_d15_zero[s_start:s_end], decoder=decoder_r3)
        residual_errors_b4, error_mask_bit = bit_flip_EC_block(residual_errors_b3, a4_d15_plus[s_start:s_end], decoder=decoder_r3)
        total_num_errors += np.logical_or(error_mask_phase, error_mask_bit).astype(int).sum()
        total_Z_errors += sum(error_mask_phase)
        total_X_errors += sum(error_mask_bit)
        print(f"#errors/#samples: {total_num_errors}/{s_end}")
        print(f"#phase errors/#samples: {total_Z_errors}/{s_end}, #bit errors/#samples: {total_X_errors}/{s_end}")
        print(f"error rate: {total_num_errors/(s_end)}")
    end = time.time()
    print(f"Total elasped time {end-start} seconds.")

def simulate_CNOT_rectangle(num_batch=200): # extended rectangle for logical transversal CNOT between two blocks
    ''' -------X--------------.------corr----.----------X--------------.-------
    CONTROL    |              |       ||     |          |              |
       ca1 |0> .--MX  ca2 |+> X--MZ ===      |  ca3 |0> .--MX  ca4 |+> X--MZ
                                             |
        -------X--------------.------corr----X----------X--------------.-------
    TARGET     |              |       ||                |              |
       ta1 |0> .--MX  ta1 |+> X--MZ ===         ta3 |0> .--MX  ta4 |+> X--MZ
    '''
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_num_errors_control = 0; total_num_errors_target = 0
    num_shots = num_batch * bs
    start = time.time()
    ca1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 0)
    ca2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 0)
    ca3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 1)
    ca4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 1)
    ta1_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 2)
    ta2_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 2)
    ta3_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 3)
    ta4_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 3)
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1_control = phase_flip_EC_block(ancilla_errors=ca1_d15_zero[s_start:s_end])
        residual_errors_b2_control = bit_flip_EC_block(residual_errors_b1_control, ca2_d15_plus[s_start:s_end], disable_correction_error=True)
        residual_errors_b1_target = phase_flip_EC_block(ancilla_errors=ta1_d15_zero[s_start:s_end])
        residual_errors_b2_target = bit_flip_EC_block(residual_errors_b1_target, ta2_d15_plus[s_start:s_end], disable_correction_error=True)
        residual_errors_control, residual_errors_target = logical_CNOT(residual_errors_b2_control, residual_errors_b2_target)
        residual_errors_b3_control, error_mask_phase_control = phase_flip_EC_block(residual_errors_control, ca3_d15_zero[s_start:s_end], decoder=decoder_r3)
        residual_errors_b4_control, error_mask_bit_control = bit_flip_EC_block(residual_errors_b3_control, ca4_d15_plus[s_start:s_end], decoder=decoder_r3)
        residual_errors_b3_target, error_mask_phase_target = phase_flip_EC_block(residual_errors_target, ta3_d15_zero[s_start:s_end], decoder=decoder_r3)
        residual_errors_b4_target, error_mask_bit_target = bit_flip_EC_block(residual_errors_b3_target, ta4_d15_plus[s_start:s_end], decoder=decoder_r3)
        control_errors = np.logical_or(error_mask_phase_control, error_mask_bit_control)
        target_errors = np.logical_or(error_mask_phase_target, error_mask_bit_target)
        total_num_errors += np.logical_or(control_errors, target_errors).astype(int).sum()
        total_num_errors_control += sum(control_errors)
        total_num_errors_target += sum(target_errors)
        print(f"#errors/#samples: {total_num_errors}/{s_end}")
        print(f"#control errors/#samples: {total_num_errors_control}/{s_end}, #phase flip: {sum(error_mask_phase_control)}, #bit flip: {sum(error_mask_bit_control)}")
        print(f"#target errors/#samples: {total_num_errors_target}/{s_end}, #phase flip: {sum(error_mask_phase_target)}, #bit flip: {sum(error_mask_bit_target)}")
        print(f"error rate: {total_num_errors/(s_end)}")
    end = time.time()
    print(f"Total elasped time {end-start} seconds.")


def simulate_CNOT_rectangle_bit_first(num_batch=200): # extended rectangle for logical transversal CNOT between two blocks
    ''' -------.--------------X------corr----.----------.--------------X-------
    CONTROL    |              |       ||     |          |              |
       ca1 |+> X--MZ  ca2 |0> .--MX ===      |  ca3 |+> X--MZ  ca4 |0> .--MX
                                             |
        -------.--------------X------corr----X----------.--------------X-------
    TARGET     |              |       ||                |              |
       ta1 |+> X--MZ  ta1 |0> .--MX ===         ta3 |+> X--MZ  ta4 |0> .--MX
    '''
    decoder_r3 = PyDecoder_polar_SCL(3)
    total_num_errors = 0
    total_num_errors_control = 0; total_num_errors_target = 0
    num_shots = num_batch * bs
    start = time.time()
    ca1_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 0)
    ca2_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 0)
    ca3_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 1)
    ca4_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 1)
    ta1_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 2)
    ta2_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 2)
    ta3_d15_plus = sample_ancilla_error(num_shots, 15, 'plus', 3)
    ta4_d15_zero = sample_ancilla_error(num_shots, 15, 'zero', 3)
    for round in range(num_batch):
        s_start = round * bs
        s_end = (round+1) * bs
        residual_errors_b1_control = bit_flip_EC_block(ancilla_errors=ca1_d15_plus[s_start:s_end])
        residual_errors_b2_control = phase_flip_EC_block(residual_errors_b1_control, ca2_d15_zero[s_start:s_end], disable_correction_error=True)
        residual_errors_b1_target = bit_flip_EC_block(ancilla_errors=ta1_d15_plus[s_start:s_end])
        residual_errors_b2_target = phase_flip_EC_block(residual_errors_b1_target, ta2_d15_zero[s_start:s_end], disable_correction_error=True)
        residual_errors_control, residual_errors_target = logical_CNOT(residual_errors_b2_control, residual_errors_b2_target)
        residual_errors_b3_control, error_mask_bit_control = bit_flip_EC_block(residual_errors_control, ca3_d15_plus[s_start:s_end], decoder=decoder_r3)
        residual_errors_b4_control, error_mask_phase_control = phase_flip_EC_block(residual_errors_b3_control, ca4_d15_zero[s_start:s_end], decoder=decoder_r3)
        residual_errors_b3_target, error_mask_bit_target = bit_flip_EC_block(residual_errors_target, ta3_d15_plus[s_start:s_end], decoder=decoder_r3)
        residual_errors_b4_target, error_mask_phase_target = phase_flip_EC_block(residual_errors_b3_target, ta4_d15_zero[s_start:s_end], decoder=decoder_r3)
        control_errors = np.logical_or(error_mask_phase_control, error_mask_bit_control)
        target_errors = np.logical_or(error_mask_phase_target, error_mask_bit_target)
        total_num_errors += np.logical_or(control_errors, target_errors).astype(int).sum()
        total_num_errors_control += sum(control_errors)
        total_num_errors_target += sum(target_errors)
        print(f"#errors/#samples: {total_num_errors}/{s_end}")
        print(f"#control errors/#samples: {total_num_errors_control}/{s_end}, #phase flip: {sum(error_mask_phase_control)}, #bit flip: {sum(error_mask_bit_control)}")
        print(f"#target errors/#samples: {total_num_errors_target}/{s_end}, #phase flip: {sum(error_mask_phase_target)}, #bit flip: {sum(error_mask_bit_target)}")
        print(f"error rate: {total_num_errors/(s_end)}")
    end = time.time()
    print(f"Total elasped time {end-start} seconds.")


if __name__ == "__main__":
    # simulate_single_qubit_Clifford_rectangle(gate_type='H')
    # simulate_single_qubit_Clifford_rectangle(gate_type='S')
    # simulate_code_switching_rectangle()
    # simulate_CNOT_rectangle()
    simulate_CNOT_rectangle_bit_first()