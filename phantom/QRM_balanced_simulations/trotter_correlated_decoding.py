import stim
print(stim.__version__)
import numpy as np
import time, sys, argparse
from typing import List
from functools import reduce
from collections import Counter
from utils import z_component, x_component, lists_to_pauli_string, AncillaErrorLoader, propagate, form_pauli_string, MeasurementCircuit, pauli_string_is_all_equal, format_pauli_string
import settings
from settings import int2bin, bin2int, bin_wt, append_hypercube_encoding_circuit, append_initialization, append_measurement
sys.path.append("../")
from PyDecoder_polar import PyDecoder_polar_SCL
from src.simulation_v2.decoder_mle import *

m = settings.m
N = settings.N
l = settings.l
K = settings.K
d = settings.d

code = settings.code
Lx = settings.Lx
Lz = settings.Lz
logical_indices = settings.logical_indices
low_wt_Hx = settings.low_wt_Hx
low_wt_Hz = settings.low_wt_Hz

X_stab_detector_circuit_str = ""
for i, s in enumerate(low_wt_Hx):
    nnz = np.nonzero(s)[0]
    det_str = "DETECTOR"
    for ind in nnz:
        det_str += f" rec[{-N+ind}]"
    det_str += "\n"
    X_stab_detector_circuit_str += det_str
X_stab_detector_circuit = stim.Circuit(X_stab_detector_circuit_str)

# compute Z stabilizer detector string
# right now not XORing with previous round
Z_stab_detector_circuit_str = ""
for i, s in enumerate(low_wt_Hz):
    nnz = np.nonzero(s)[0]
    det_str = "DETECTOR"
    for ind in nnz:
        det_str += f" rec[{-N+ind}]"
    det_str += "\n"
    Z_stab_detector_circuit_str += det_str
Z_stab_detector_circuit = stim.Circuit(Z_stab_detector_circuit_str)

##################################################################################

def get_CNOT_window_MLE_setup():

    print(f"get_CNOT_window_MLE_setup: p_CNOT={p_CNOT}, p_meas={p_meas}, p_reset={p_reset}, p_idle={p_idle}, p_residual={p_residual}, p_prep={p_prep}")

    tick_circuits = [] # for PauliString.after

    # create stim circuit for exRec (state prep are perfect + residual error modeled as iid flip)
    circuit = stim.Circuit()
    # data patch 1, in 0000
    append_initialization(circuit, "0000", 0)
    append_hypercube_encoding_circuit(circuit, 0)
    # ancilla patch for data patch 1, first in 0000 for phase EC
    append_initialization(circuit, "0000", N)
    append_hypercube_encoding_circuit(circuit, N)
    # data patch 2, in 0000
    append_initialization(circuit, "0000", 2*N)
    append_hypercube_encoding_circuit(circuit, 2*N)
    # ancilla patch for data patch 2, first in 0000 for phase EC
    append_initialization(circuit, "0000", 3*N)
    append_hypercube_encoding_circuit(circuit, 3*N)


    # residual error on all patches
    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_residual)
        circuit.append("DEPOLARIZE1", N+i, p_prep)
        circuit.append("DEPOLARIZE1", 2*N+i, p_residual)
        circuit.append("DEPOLARIZE1", 3*N+i, p_prep)
    # TICK 0
    circuit.append("TICK")
    tick_circuit  = stim.Circuit()
    # Leading Steane EC
    # first, phase EC
    for i in range(N):
        circuit.append("CNOT", [i+N, i])
        tick_circuit.append("CNOT", [i+N, i])
        circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)
        circuit.append("CNOT", [i+3*N, i+2*N])
        tick_circuit.append("CNOT", [i+3*N, i+2*N])
        circuit.append("DEPOLARIZE2", [i+3*N, i+2*N], p_CNOT)
    # TICK 1
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("Z_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("Z_ERROR", i+3*N, p_meas)
    # TICK 2
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MX", i+N)
        tick_circuit.add_MX(i+N)
    circuit += X_stab_detector_circuit
    for i in range(N):
        circuit.append("MX", i+3*N)
        tick_circuit.add_MX(i+3*N)
    circuit += X_stab_detector_circuit
    # TICK 3
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # next, bit EC
    # re-initialize the ancilla
    append_initialization(circuit, "++++", N)
    append_hypercube_encoding_circuit(circuit, N)
    append_initialization(circuit, "++++", 3*N)
    append_hypercube_encoding_circuit(circuit, 3*N)

    for i in range(N):
        circuit.append("DEPOLARIZE1", N+i, p_prep)
        circuit.append("DEPOLARIZE1", 3*N+i, p_prep)
    # TICK 4
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")


    tick_circuit  = stim.Circuit()
    for i in range(N):
        circuit.append("CNOT", [i, i+N])
        tick_circuit.append("CNOT", [i, i+N])
        circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)
        circuit.append("CNOT", [i+2*N, i+3*N])
        tick_circuit.append("CNOT", [i+2*N, i+3*N])
        circuit.append("DEPOLARIZE2", [i+2*N, i+3*N], p_CNOT)
    # TICK 5
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("X_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("X_ERROR", i+3*N, p_meas)
    # TICK 6
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MZ", i+N)
        tick_circuit.add_MZ(i+N)
    circuit += Z_stab_detector_circuit
    for i in range(N):
        circuit.append("MZ", i+3*N)
        tick_circuit.add_MZ(i+3*N)
    circuit += Z_stab_detector_circuit
    # TICK 7
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    tick_circuit = stim.Circuit()
    # the CNOT
    for i in range(N):
        circuit.append("CNOT", [i, i+2*N])
        tick_circuit.append("CNOT", [i, i+2*N])
        circuit.append("DEPOLARIZE2", [i, i+2*N], p_CNOT)
    # TICK 8
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # Trailing Steane EC
    # first, phase EC
    # re-initialize the ancilla
    append_initialization(circuit, "0000", N)
    append_hypercube_encoding_circuit(circuit, N)
    append_initialization(circuit, "0000", 3*N)
    append_hypercube_encoding_circuit(circuit, 3*N)

    for i in range(N):
        circuit.append("DEPOLARIZE1", N+i, p_prep)
        circuit.append("DEPOLARIZE1", 3*N+i, p_prep)
    # TICK 9
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = stim.Circuit()
    for i in range(N):
        circuit.append("CNOT", [i+N, i])
        tick_circuit.append("CNOT", [i+N, i])
        circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)
        circuit.append("CNOT", [i+3*N, i+2*N])
        tick_circuit.append("CNOT", [i+3*N, i+2*N])
        circuit.append("DEPOLARIZE2", [i+3*N, i+2*N], p_CNOT)
    # TICK 10
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("Z_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("Z_ERROR", i+3*N, p_meas)
    # TICK 11
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MX", i+N)
        tick_circuit.add_MX(i+N)
    circuit += X_stab_detector_circuit
    for i in range(N):
        circuit.append("MX", i+3*N)
        tick_circuit.add_MX(i+3*N)
    circuit += X_stab_detector_circuit
    # TICK 12
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # next, bit EC
    # re-initialize the ancilla
    append_initialization(circuit, "++++", N)
    append_hypercube_encoding_circuit(circuit, N)
    append_initialization(circuit, "++++", 3*N)
    append_hypercube_encoding_circuit(circuit, 3*N)

    for i in range(N):
        circuit.append("DEPOLARIZE1", N+i, p_prep)
        circuit.append("DEPOLARIZE1", 3*N+i, p_prep)
    # TICK 13
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = stim.Circuit()
    for i in range(N):
        circuit.append("CNOT", [i, i+N])
        tick_circuit.append("CNOT", [i, i+N])
        circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)
        circuit.append("CNOT", [i+2*N, i+3*N])
        tick_circuit.append("CNOT", [i+2*N, i+3*N])
        circuit.append("DEPOLARIZE2", [i+2*N, i+3*N], p_CNOT)
    # TICK 14
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("X_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("X_ERROR", i+3*N, p_meas)
    # TICK 15
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MZ", i+N)
        tick_circuit.add_MZ(i+N)
    circuit += Z_stab_detector_circuit
    for i in range(N):
        circuit.append("MZ", i+3*N)
        tick_circuit.add_MZ(i+3*N)
    circuit += Z_stab_detector_circuit
    # TICK 16
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # diagram = circuit.diagram('timeline-svg')
    # with open(f'exRec.svg', 'w') as f:
    #     print(diagram, file=f)

    # dem = circuit.detector_error_model()
    # chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)

    # print(col_dict)

    dem: stim.DetectorErrorModel = circuit.detector_error_model()
    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    flat_error_instructions: List[stim.DemInstruction] = [
        instruction
        for instruction in dem.flattened()
        if instruction.type == 'error'
    ]

    num_faults = len(flat_error_instructions)
    num_detectors = circuit.num_detectors
    print("length of dem.flattened()", len(dem.flattened()))
    print(f"number of faults {num_faults}, number of detectors {num_detectors} -- shape of the DEM matrix")
    print("number of ticks in the circuit", circuit.num_ticks)
    print("tick_circuits length", len(tick_circuits))

    # generate propagation dictionary.
    start = time.time()
    dem_check_matrix = np.zeros((num_detectors, num_faults), dtype=np.uint8)
    priors = np.zeros((num_faults), dtype=np.double)
    prop_dict = {}
    key_set = set()
    fault_explanation_dict = {}
    for i in range(len(flat_error_instructions)):

        dets = []
        instruction = flat_error_instructions[i]
        priors[i] = instruction.args_copy()[0]
        for t in instruction.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
        key = " ".join([f"D{s}" for s in sorted(dets)])
        # print(f"instruction i={i} probability={priors[i]} triggers detectors {key}")
        key_set.add(key)
        for det_id in dets:
            dem_check_matrix[det_id, i] = 1

        dem_filter = stim.DetectorErrorModel()
        dem_filter.append(flat_error_instructions[i])
        explained_errors: List[stim.ExplainedError] = circuit.explain_detector_error_model_errors(dem_filter=dem_filter, reduce_to_one_representative_error=False)
        tick_list = []
        final_pauli_string_control_all = []
        final_pauli_string_target_all = []
        fault_explanation_dict[i] = explained_errors[0].circuit_error_locations[0]
        for rep_loc in explained_errors[0].circuit_error_locations:
            # print(rep_loc)
            tick = rep_loc.tick_offset
            tick_list.append(tick)
            # for sanity checks
            final_pauli_string = propagate(form_pauli_string(rep_loc.flipped_pauli_product, 4*N), tick_circuits[tick:])
            final_pauli_string_control = final_pauli_string[:N]
            final_pauli_string_target = final_pauli_string[2*N:3*N]
            final_pauli_string_control_all.append(final_pauli_string_control)
            final_pauli_string_target_all.append(final_pauli_string_target)
            # print(f"fault at tick={tick}, on control {format_pauli_string(final_pauli_string_control)}; on target {format_pauli_string(final_pauli_string_target)}")

            # actual things to be stored in the propagation dictionary
            pauli_string_before_CNOT = propagate(form_pauli_string(rep_loc.flipped_pauli_product, 4*N), tick_circuits[tick:7])
            pauli_string_control_before_CNOT = pauli_string_before_CNOT[:N]
            pauli_string_target_before_CNOT = pauli_string_before_CNOT[2*N:3*N]
            # print(f"fault at tick={tick}, just before the CNOT, on control {format_pauli_string(pauli_string_control_before_CNOT)}; on target {format_pauli_string(pauli_string_target_before_CNOT)}")
        if (np.array(tick_list) < 7).all(): # before the transversal CNOT
            # check residual Pauli string all equal
            assert pauli_string_is_all_equal(final_pauli_string_control_all)
            assert pauli_string_is_all_equal(final_pauli_string_target_all)
            # in sliding window decoding, only commit to those faults
            prop_dict[i] = (pauli_string_control_before_CNOT, pauli_string_target_before_CNOT)
            assert (pauli_string_control_before_CNOT + pauli_string_target_before_CNOT).weight <= 1

    end = time.time()
    print(f"Total Elapsed time: {end-start}")   
    print("unique key (triggered detector string)", len(key_set))
    print("faults in prop_dict (commit to in sliding window)", len(prop_dict.keys()))

    dem_check_matrix_indices = [np.nonzero(row)[0] for row in dem_check_matrix]
    return dem_check_matrix_indices, priors, prop_dict

############################## for correlated list decoding in each window ######################
def correlated_phase_flip_EC(before, after, name=""):
    correction = []
    for i in range(num_shots):
        nnz1 = list(np.nonzero(before[i])[0])
        nnz2 = list(np.nonzero(after[i])[0])
        decoder.correlated_decode_Z_flip(nnz1, nnz2)
        corr = np.zeros(N, dtype=np.uint8)
        for idx in decoder.correlated_Z_correction: corr[idx] = 1
        correction.append(corr)
        # if len(decoder.correlated_Z_correction) > 1:
        #     print(f"x1 i={i} corr {decoder.correlated_Z_correction}")
    return np.array(correction)


def correlated_bit_flip_EC(before, after, name=""):
    correction = []
    for i in range(num_shots):
        nnz1 = list(np.nonzero(before[i])[0])
        nnz2 = list(np.nonzero(after[i])[0])
        decoder.correlated_decode_X_flip(nnz1, nnz2)
        corr = np.zeros(N, dtype=np.uint8)
        for idx in decoder.correlated_X_correction: corr[idx] = 1
        correction.append(corr)
        # if len(decoder.correlated_X_correction) > 1:
        #     print(f"x1 i={i} corr {decoder.correlated_X_correction}")
    return np.array(correction)
#################################################################################################
###################### usual list decoder to handle the boundary (noiseless SE) #################
def phase_flip_EC(e, name=""):
    wt_4_cnt = 0
    correction = []
    for i in range(num_shots):
        num_flip = decoder.decode_Z_flip(list(np.nonzero(e[i])[0]))
        # if num_flip > 3: print(f"{name} i={i} num Z flip: {num_flip}")
        if num_flip > 3: wt_4_cnt += 1
        corr = np.zeros(N, dtype=np.uint8)
        for i in decoder.Z_correction: corr[i] = 1
        correction.append(corr)
    print(f"phase EC {name}: decoder seeing wt>4 cnt", wt_4_cnt)
    return np.array(correction)

def bit_flip_EC(e, name=""):
    wt_4_cnt = 0
    correction = []
    for i in range(num_shots):
        num_flip = decoder.decode_X_flip(list(np.nonzero(e[i])[0]))
        # if num_flip > 3: print(f"{name} i={i} num X flip: {num_flip}")
        if num_flip > 3: wt_4_cnt += 1
        corr = np.zeros(N, dtype=np.uint8)
        for i in decoder.X_correction: corr[i] = 1
        correction.append(corr)
    print(f"bit EC {name}: decoder seeing wt>4 cnt", wt_4_cnt)
    return np.array(correction)
#################################################################################################

def add_Steane(sim, Steane_cnt):
    zero_prep_circuit = stim.Circuit()
    for i in range(num_block):
        append_initialization(zero_prep_circuit, "0000", (i+num_block)*N)
        append_hypercube_encoding_circuit(zero_prep_circuit, (i+num_block)*N)

    sim.do(zero_prep_circuit)

    # inject residual errors on |0000> and |++++> patches
    loader = AncillaErrorLoader()
    ancilla_zero_block_residual_errors = []
    ancilla_plus_block_residual_errors = []
    for i in range(num_block):
        a_zero = loader.sample_ancilla_error(num_shots, Steane_cnt, f"logs_prep_zero/{dir_error_rate}")
        a_plus = loader.sample_ancilla_error(num_shots, Steane_cnt, f"logs_prep_plus/{dir_error_rate}")
        ancilla_zero_block_residual_errors.append( loader.process_ancilla_error(a_zero, 'zero') )
        ancilla_plus_block_residual_errors.append( loader.process_ancilla_error(a_plus, 'plus') )
        Steane_cnt += 1

    data_ancilla_zero_residual_errors = []
    for i in range(num_shots):
        e = stim.PauliString(num_block * N) # no error to inject on data blocks
        for block in ancilla_zero_block_residual_errors:
            e += block[i]
        data_ancilla_zero_residual_errors.append(e)

    data_ancilla_plus_residual_errors = []
    for i in range(num_shots):
        e = stim.PauliString(num_block * N) # no error to inject on data blocks
        for block in ancilla_plus_block_residual_errors:
            e += block[i]
        data_ancilla_plus_residual_errors.append(e)

    X_component, Z_component = np.array([e.to_numpy() for e in data_ancilla_zero_residual_errors]).transpose(1,2,0) # each shape (2*num_block*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    zero_coupling_circuit = stim.Circuit()
    offset = N*num_block
    for i in range(offset):
        zero_coupling_circuit.append("CNOT", [i+offset, i])
        zero_coupling_circuit.append("DEPOLARIZE2", [i+offset, i], p_CNOT)

    for i in range(offset):
        zero_coupling_circuit.append("Z_ERROR", i+offset, p_meas)
        zero_coupling_circuit.append("MX", i+offset)

    sim.do(zero_coupling_circuit)
    # phase_EC_measurement_result = sim.get_measurement_flips()
    # print("phase EC measurement result shape", phase_EC_measurement_result.shape)

    plus_prep_circuit = stim.Circuit()
    for i in range(num_block):
        append_initialization(plus_prep_circuit, "++++", (i+num_block)*N)
        append_hypercube_encoding_circuit(plus_prep_circuit, (i+num_block)*N)

    sim.do(plus_prep_circuit)

    X_component, Z_component = np.array([e.to_numpy() for e in data_ancilla_plus_residual_errors]).transpose(1,2,0) # each shape (2*num_block*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)
    
    plus_coupling_circuit = stim.Circuit()
    for i in range(offset):
        plus_coupling_circuit.append("CNOT", [i, i+offset])
        plus_coupling_circuit.append("DEPOLARIZE2", [i, i+offset], p_CNOT)

    for i in range(offset):
        plus_coupling_circuit.append("X_ERROR", i+offset, p_meas)
        plus_coupling_circuit.append("MZ", i+offset)

    sim.do(plus_coupling_circuit)
    # bit_EC_measurement_result = sim.get_measurement_flips()
    # print("bit EC measurement result shape", bit_EC_measurement_result.shape)

    # return phase_EC_measurement_result, bit_EC_measurement_result
    return Steane_cnt

def simulate_trotter_circuit(num_block, depth, index, decoder_method):

    print(f"simulate_trotter_circuit: p_CNOT={p_CNOT}, p_meas={p_meas}, p_reset={p_reset}, p_idle={p_idle}, #blocks={num_block}, index={index}, depth={depth}, #shots={num_shots}")
    init_str = "+0" * (num_block//2)
    measurement_str = "+0" * (num_block//2)
    CNOT = []
    layer_one_CNOT = []
    for i in range(num_block//2):
        layer_one_CNOT.append((2*i, 2*i+1))
    layer_two_CNOT = []
    for i in range(num_block//2):
        layer_two_CNOT.append((2*i+1, (2*i+2)%num_block))

    CNOT = [layer_one_CNOT, layer_one_CNOT, layer_two_CNOT, layer_two_CNOT]
    CNOT = CNOT * depth

    # circuit description (on the block level)
    logical_circuit_block_level = {
        "initialization": "+0+0",
        "CNOT": [ [(0,1), (2,3)], [(0,1), (2,3)], [(1,2), (3,0)], [(1,2), (3,0)] ] * 8,
        "measurement": "+0+0"
    }

    logical_circuit_block_level["initialization"] = init_str
    logical_circuit_block_level["CNOT"] = CNOT
    logical_circuit_block_level["measurement"] = init_str
    print(logical_circuit_block_level)

    ########################## sanity test ###############################
    init_str = logical_circuit_block_level["initialization"]
    logical_circuit = stim.Circuit()
    for i, state in enumerate(init_str):
        if state == '0':
            logical_circuit.append("RZ", i)
        elif state == '+':
            logical_circuit.append("RX", i)

    CNOT = logical_circuit_block_level["CNOT"]
    for list_tup in CNOT:
        for (ctrl, targ) in list_tup:
            logical_circuit.append("CNOT", [ctrl, targ])

    measurement_str = logical_circuit_block_level["measurement"]
    for i, state in enumerate(measurement_str):
        if state == '0':
            logical_circuit.append("MZ", i)
        elif state == '+':
            logical_circuit.append("MX", i)

    s = logical_circuit.compile_sampler()
    result = s.sample(shots=100)
    assert not result.any(), "trotter logical circuit test: not all zero!"
    ########################## sanity test end ##########################


    init_str = logical_circuit_block_level["initialization"]
    state_prep_circuit = stim.Circuit()
    for i, state in enumerate(init_str):
        print(f"i={i}, state={state}")
        append_initialization(state_prep_circuit, state*4, N*i)
        append_hypercube_encoding_circuit(state_prep_circuit, N*i)

    sim = stim.FlipSimulator(batch_size=num_shots, num_qubits=2*num_block*N)
    sim.do(state_prep_circuit)

    loader = AncillaErrorLoader()
    data_block_init_residual_errors = []
    plus_Steane_cnt = index
    zero_Steane_cnt = index
    for i, state in enumerate(init_str):
        if state == '0':
            a_zero = loader.sample_ancilla_error(num_shots, zero_Steane_cnt, f"logs_prep_zero/{dir_error_rate}")
            data_block_init_residual_errors.append( loader.process_ancilla_error(a_zero, 'zero') )
            zero_Steane_cnt += 1
        elif state == '+':
            a_plus = loader.sample_ancilla_error(num_shots, plus_Steane_cnt, f"logs_prep_plus/{dir_error_rate}")
            data_block_init_residual_errors.append( loader.process_ancilla_error(a_plus, 'plus') )
            plus_Steane_cnt += 1

    Steane_cnt = max(plus_Steane_cnt, zero_Steane_cnt)
    print("Steane cnt before all this", Steane_cnt)

    data_block_init_residual_errors_total = []
    for i in range(num_shots):
        e = stim.PauliString()
        for block in data_block_init_residual_errors:
            e += block[i]
        data_block_init_residual_errors_total.append(e)

    X_component, Z_component = np.array([e.to_numpy() for e in data_block_init_residual_errors_total]).transpose(1,2,0) # each shape (num_block*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)


    CNOT = logical_circuit_block_level["CNOT"]
    for tick, list_tup in enumerate(CNOT):
        slice_circuit = stim.Circuit()
        for (ctrl, targ) in list_tup:
            for i in range(N):
                slice_circuit.append("CNOT", [ctrl*N+i, targ*N+i])
                slice_circuit.append("DEPOLARIZE2", [ctrl*N+i, targ*N+i], p_CNOT)
        sim.do(slice_circuit)
        Steane_cnt = add_Steane(sim, Steane_cnt)

    measurement_str = logical_circuit_block_level["measurement"]
    measurement_circuit = stim.Circuit()
    for i, state in enumerate(measurement_str):
        if state == "0":
            for idx in range(N):
                measurement_circuit.append("X_ERROR", i*N+idx, p_meas)
                measurement_circuit.append("MZ", i*N+idx, p_meas)
        elif state == "+":
            for idx in range(N):
                measurement_circuit.append("Z_ERROR", i*N+idx, p_meas)
                measurement_circuit.append("MX", i*N+idx, p_meas)
    sim.do(measurement_circuit)            

    measurement_result = sim.get_measurement_flips().T
    measurement_result = measurement_result.astype(np.uint8)
    print("measurement result shape", measurement_result.shape)
    num_measurement = measurement_result.shape[1]

    def bit_get_result(tick, idx):
        start = tick * (2*num_block) + num_block + idx
        print(f"bit start={start}, mod num_block = {start % num_block}")
        start *= N
        return measurement_result[:, start:start+N]

    def phase_get_result(tick, idx):
        start = tick * (2*num_block) + idx
        print(f"phase start={start}, mod num_block = {start % num_block}")
        start *= N
        return measurement_result[:, start:start+N]

    def bit_update_result(tick, idx, corr):
        start = tick * (2*num_block) + num_block + idx
        print(f"bit update result at start={start}, mod num_block = {start % num_block}")
        start *= N
        measurement_result[:, start:start+N] = (measurement_result[:, start:start+N] + corr) % 2

    def phase_update_result(tick, idx, corr):
        start = tick * (2*num_block) + idx
        print(f"phase update result at start={start}, mode num_block = {start % num_block}")
        start *= N
        measurement_result[:, start:start+N] = (measurement_result[:, start:start+N] + corr) % 2
        
    logical_circuit_tick = []
    for list_tup in CNOT:
        c = stim.Circuit()
        for (ctrl, targ) in list_tup:
            c.append("CNOT", [ctrl, targ])
        logical_circuit_tick.append(c)

    def propagate_and_update_X_correction(tick, idx, corr):
        e = lists_to_pauli_string([idx], [], num_block)
        for tick_offset, c in enumerate(logical_circuit_tick[tick+1:]):
            temp_tick = tick + tick_offset + 1
            e = e.after(c)
            print(f"propagate X: tick_offset={tick_offset}, e={e}")
            x, _ = e.to_numpy()
            for temp_idx in np.nonzero(x)[0]:
                bit_update_result(temp_tick, temp_idx, corr)
        # update final transversal measurement result (only update MZ)
        x, _ = e.to_numpy()
        for temp_idx in np.nonzero(x)[0]:
            if measurement_str[temp_idx] == '0':
                print(f"propagate X: update final transveral measurement at block {temp_idx}")
                offset = num_measurement - num_block*N + temp_idx*N
                measurement_result[:, offset:offset+N] = (measurement_result[:, offset:offset+N] + corr) % 2

    def propagate_and_update_Z_correction(tick, idx, corr):
        e = lists_to_pauli_string([], [idx], num_block)
        for tick_offset, c in enumerate(logical_circuit_tick[tick+1:]):
            temp_tick = tick + tick_offset + 1
            e = e.after(c)
            print(f"propagate Z: tick_offset={tick_offset}, e={e}")
            _, z = e.to_numpy()
            for temp_idx in np.nonzero(z)[0]:
                phase_update_result(temp_tick, temp_idx, corr)
        # update final transversal measurement result (only update MX)
        _, z = e.to_numpy()
        for temp_idx in np.nonzero(z)[0]:
            if measurement_str[temp_idx] == '+':
                print(f"propagate Z: update final transveral measurement at block {temp_idx}")
                offset = num_measurement - num_block*N + temp_idx*N
                measurement_result[:, offset:offset+N] = (measurement_result[:, offset:offset+N] + corr) % 2

    start = time.time()
    if decoder_method == 'list':
        for tick, list_tup in enumerate(CNOT[1:]):
            for (ctrl, targ) in list_tup:
                ctrl_bit_LEC = bit_get_result(tick, ctrl)
                ctrl_bit_TEC = bit_get_result(tick+1, ctrl)
                targ_phase_LEC = phase_get_result(tick, targ)
                targ_phase_TEC = phase_get_result(tick+1, targ)
                ctrl_bit_corr = correlated_bit_flip_EC(ctrl_bit_LEC, ctrl_bit_TEC)
                print("average weight of ctrl_bit_corr", ctrl_bit_corr.sum(axis=1).mean())
                targ_phase_corr = correlated_phase_flip_EC(targ_phase_LEC, targ_phase_TEC)
                print("average weight of targ_phase_corr", targ_phase_corr.sum(axis=1).mean())
                # propagate the correction
                propagate_and_update_X_correction(tick, ctrl, ctrl_bit_corr)
                propagate_and_update_Z_correction(tick, targ, targ_phase_corr)

                targ_bit_LEC = bit_get_result(tick, targ)
                targ_bit_TEC = bit_get_result(tick+1, targ)
                ctrl_phase_LEC = phase_get_result(tick, ctrl)
                ctrl_phase_TEC = phase_get_result(tick+1, ctrl)
                targ_bit_corr = correlated_bit_flip_EC(targ_bit_LEC, targ_bit_TEC)
                print("average weight of targ_bit_corr", targ_bit_corr.sum(axis=1).mean())
                ctrl_phase_corr = correlated_phase_flip_EC(ctrl_phase_LEC, ctrl_phase_TEC)
                print("average weight of ctrl_phase_corr", ctrl_phase_corr.sum(axis=1).mean())
                # propagate the correction
                propagate_and_update_X_correction(tick, targ, targ_bit_corr)
                propagate_and_update_Z_correction(tick, ctrl, ctrl_phase_corr)

    elif decoder_method == 'MLE':

        dem_check_matrix_indices, priors, prop_dict = get_CNOT_window_MLE_setup()

        def commit_to_faults_and_get_correction(fault_list):
            # fault_list contains a list of fault indices that happened according to the decoder
            # use prop_dict to combine them
            final_pauli_string_control_all = []
            final_pauli_string_target_all = []
            for index in fault_list:
                if index in prop_dict.keys():
                    (c, t) = prop_dict[index]
                    final_pauli_string_control_all.append(c)
                    final_pauli_string_target_all.append(t)
                    # print(f"combining fault {index}, explanation {fault_explanation_dict[index]}, on control {format_pauli_string(c)}; on target {format_pauli_string(t)}")
            final_pauli_string_control = reduce(stim.PauliString.__mul__, final_pauli_string_control_all, stim.PauliString(N))
            final_pauli_string_target = reduce(stim.PauliString.__mul__, final_pauli_string_target_all, stim.PauliString(N))
            # print(f"combined fault, on control {format_pauli_string(final_pauli_string_control)}; on target {format_pauli_string(final_pauli_string_target)}")
            return final_pauli_string_control, final_pauli_string_target


        def mle_decoder_get_residual_error(detector_syndrome):
            faults, _ = mle_decoder_gurobi_decode_using_check_matrix(
                check_matrix_error_indices=dem_check_matrix_indices,
                logical_matrix_error_indices=None,
                error_probabilities=priors,
                detector_shots=detector_syndrome, # (num_shots, num_detectors)
                gurobi_n_threads=1
            )
            ctrl_bit_corr_all = []
            ctrl_phase_corr_all = []
            targ_bit_corr_all = []
            targ_phase_corr_all = []
            for fault_list in faults:
                ctrl_corr, targ_corr = commit_to_faults_and_get_correction(fault_list)
                ctrl_bit_corr, ctrl_phase_corr = ctrl_corr.to_numpy()
                targ_bit_corr, targ_phase_corr = targ_corr.to_numpy()
                ctrl_bit_corr_all.append(ctrl_bit_corr)
                ctrl_phase_corr_all.append(ctrl_phase_corr)
                targ_bit_corr_all.append(targ_bit_corr)
                targ_phase_corr_all.append(targ_phase_corr)
            return np.array(ctrl_bit_corr_all), np.array(ctrl_phase_corr_all), np.array(targ_bit_corr_all), np.array(targ_phase_corr_all)

        for tick, list_tup in enumerate(CNOT[1:]):
            for (ctrl, targ) in list_tup:
                ctrl_phase_LEC = phase_get_result(tick, ctrl)
                targ_phase_LEC = phase_get_result(tick, targ)
                ctrl_bit_LEC = bit_get_result(tick, ctrl)
                targ_bit_LEC = bit_get_result(tick, targ)
                ctrl_phase_TEC = phase_get_result(tick+1, ctrl)
                targ_phase_TEC = phase_get_result(tick+1, targ)
                ctrl_bit_TEC = bit_get_result(tick+1, ctrl)
                targ_bit_TEC = bit_get_result(tick+1, targ)
                detector_syndrome = np.hstack((
                    ctrl_phase_LEC @ low_wt_Hx.T % 2,
                    targ_phase_LEC @ low_wt_Hx.T % 2,
                    ctrl_bit_LEC @ low_wt_Hz.T % 2,
                    targ_bit_LEC @ low_wt_Hz.T % 2,
                    ctrl_phase_TEC @ low_wt_Hx.T % 2,
                    targ_phase_TEC @ low_wt_Hx.T % 2,
                    ctrl_bit_TEC @ low_wt_Hz.T % 2,
                    targ_bit_TEC @ low_wt_Hz.T % 2
                ))

                ctrl_bit_corr, ctrl_phase_corr, targ_bit_corr, targ_phase_corr = mle_decoder_get_residual_error(detector_syndrome)

                ########################### compare against correlated list decoding ########################
                # ctrl_bit_corr_list = correlated_bit_flip_EC(ctrl_bit_LEC, ctrl_bit_TEC)
                # targ_phase_corr_list = correlated_phase_flip_EC(targ_phase_LEC, targ_phase_TEC)
                # for i in range(num_shots):
                #     print(f"shot{i} MLE  ctrl bit corr at", np.where(ctrl_bit_corr[i])[0])
                #     print(f"shot{i} list ctrl bit corr at", np.where(ctrl_bit_corr_list[i])[0])
                #     print(f"shot{i} MLE  targ phase corr at", np.where(targ_phase_corr[i])[0])
                #     print(f"shot{i} list targ phase corr at", np.where(targ_phase_corr_list[i])[0])

                # propagate the correction
                propagate_and_update_X_correction(tick, ctrl, ctrl_bit_corr)
                propagate_and_update_Z_correction(tick, targ, targ_phase_corr)

                propagate_and_update_X_correction(tick, targ, targ_bit_corr)
                propagate_and_update_Z_correction(tick, ctrl, ctrl_phase_corr)

    else:
        print("Unsupported decoding method")
        exit(-1)


    log_err_all = []
    for i, state in enumerate(measurement_str):
        offset = num_measurement - num_block*N + i*N
        if state == '0':
            log_X_correction = bit_flip_EC(measurement_result[:, offset:offset+N], f"transversal Z on block {i}")
            log_X_diff = (measurement_result[:, offset:offset+N] + log_X_correction) % 2
            assert not (log_X_diff @ code.hz.T % 2).any()
            log_err = log_X_diff @ Lz.T % 2
            log_err_all.append(log_err)
        elif state == '+':
            log_Z_correction = phase_flip_EC(measurement_result[:, offset:offset+N], f"transversal X on block {i}")
            log_Z_diff = (measurement_result[:, offset:offset+N] + log_Z_correction) % 2
            assert not (log_Z_diff @ code.hx.T % 2).any()
            log_err = log_Z_diff @ Lx.T % 2
            log_err_all.append(log_err)

    log_err_all = np.hstack(log_err_all)
    print("log_err_all shape:", log_err_all.shape) # expect (num_shots, 4*num_block) 
    total_errors = log_err_all.any(axis=1).astype(int).sum()
    print(f"#shots={num_shots}, #errors={total_errors}")
    print(log_err_all.astype(int).sum(axis=0))
    end = time.time()
    print(f"Total decoding time: {end-start}")   
    # with open(f"logs_trotter/tailored_Steane/nb{num_block}_{dir_error_rate}_index{index}_depth{depth}.npy", "wb") as f:
    #     np.save(f, log_err_all)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Simulate trotter circuit for phantom QRM")
    parser.add_argument("--index", type=int, default=0, help="index of the file")
    parser.add_argument("--num_shots", type=int, default=128, help="number of shots, use a multiple of 128. Use at most 10240 for p=0.003, at most 204800 for p=0.001, at most 409600 for p=0.0005")
    parser.add_argument("--p_CNOT", type=float, help="physical error rate of CNOT")
    parser.add_argument("--depth", type=int, default=8, help="depth of the trotter circuit")
    parser.add_argument("-nb", "--num_block", type=int, default=1, help="number of blocks used in the trotter circuit")
    parser.add_argument('--method', default='list', choices=['list', 'MLE'], help='decoder methods to be used with sliding window, either list or MLE')
    args = parser.parse_args()

    num_shots = args.num_shots
    p_CNOT = args.p_CNOT 
    p_meas = 5.0 * p_CNOT / 3.0
    p_reset = p_CNOT
    p_idle = p_CNOT / 300.0
    ################ priors for MLE window ######################
    p_residual = 2 * p_CNOT 
    p_prep = 0.8 * p_CNOT 
    #############################################################
    num_block = args.num_block
    depth = args.depth
    index = args.index
    decoder_method = args.method
    ################### end of settings #########################
    print(f"in main: p_CNOT={p_CNOT}, p_meas={p_meas}, p_reset={p_reset}, p_idle={p_idle}, #blocks={num_block}, index={index}, depth={depth}, #shots={num_shots}, decoder method={decoder_method}")
    dir_error_rate = "p" + str(p_CNOT).split('.')[1]
    decoder = PyDecoder_polar_SCL(l, m) # arbitrary state decoder

    simulate_trotter_circuit(num_block, depth, index, decoder_method)

