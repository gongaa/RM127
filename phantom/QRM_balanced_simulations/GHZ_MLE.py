import stim
print(stim.__version__)
import numpy as np
from typing import List, FrozenSet, Dict
import time, sys, argparse
from functools import reduce
from utils import z_component, x_component, lists_to_pauli_string, AncillaErrorLoader, MeasurementCircuit, form_pauli_string, format_pauli_string, pauli_string_is_all_equal, propagate, decompose_into_two_involutions
import matplotlib.pyplot as plt
import settings
from settings import *
sys.path.append("../")
from PyDecoder_polar import PyDecoder_polar_SCL
from src.magic.codes_q import rank
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
# two permutations for a single CNOT between blocks
perm1 = np.eye(m, dtype=int)
perm1[2,2] = 0
perm1[2,3] = 1
perm1[3,2] = 1
perm2 = perm1.copy()
perm2[3,3] = 0
# one-block CNOT ladder permutation
layer_1 = np.eye(K, dtype=int)
layer_1[3,2] = 1
layer_2 = np.eye(K, dtype=int)
layer_2[3,1] = 1
layer_2[2,0] = 1
perm = np.eye(m, dtype=int)
perm[:K,:K] = layer_2 @ layer_1 % 2
print(perm)
perm_CNOT_ladder = [N-1-bin2int(perm @ np.array(int2bin(N-1-i)) % 2) for i in range(N)]
swaps1, swaps2 = decompose_into_two_involutions(perm_CNOT_ladder)
print("SWAPS1")
print(swaps1)
print("SWAPS2")
print(swaps2)

# compute X stabilizer detector string
# right now not XORing with previous round
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

# compute X logical observable string on the second data patch
X_log_detector_circuit_str = ""
for i, s in enumerate(Lx[1:]):
    nnz = np.nonzero(s)[0]
    det_str = f"OBSERVABLE_INCLUDE({i})"
    for ind in nnz:
        det_str += f" rec[{-N+ind}]"
    det_str += "\n"
    X_log_detector_circuit_str += det_str
X_log_detector_circuit = stim.Circuit(X_log_detector_circuit_str)

#########################################################################################
'''
Decompose the GHZ state preparation into two parts
|+000> preparation -- handled in get_plus_triple_zero_prep_setup
and the following circuit using transversal CNOT only
|+000> -- EC -- Perm --*-- EC ---*--- EC -- ...
                       |         |
|0000> ----------------X-- EC ---|*-- EC -- ...
                                 ||
|0000> --------------------------X|-- EC -- ...
                                  |
|0000> ---------------------------X-- EC -- ...
...

A common pattern each window is
? ----- EC -- Perm? --*-- EC --
                      |
|0000> ---------------X-- EC --
and DEM is constructed in get_GHZ_CNOT_window_setup
'''
#########################################################################################
def get_GHZ_CNOT_window_setup(perm=None):
    print(f"get_GHZ_CNOT_window_setup: p_CNOT={p_CNOT}, p_meas={p_meas}, p_reset={p_reset}, p_idle={p_idle}, p_residual={p_residual}, p_prep={p_prep}")

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

    # residual error on all patches
    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_residual)
        circuit.append("DEPOLARIZE1", N+i, p_prep)
        circuit.append("DEPOLARIZE1", 2*N+i, p_prep)
    # TICK 0
    circuit.append("TICK")
    tick_circuit  = stim.Circuit()
    # Leading Steane EC
    # first, phase EC
    for i in range(N):
        circuit.append("CNOT", [i+N, i])
        tick_circuit.append("CNOT", [i+N, i])
        circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)
    # TICK 1
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("Z_ERROR", i+N, p_meas)
    # TICK 2
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MX", i+N)
        tick_circuit.add_MX(i+N)
    circuit += X_stab_detector_circuit
    # TICK 3
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # next, bit EC
    # re-initialize the ancilla
    append_initialization(circuit, "++++", N)
    append_hypercube_encoding_circuit(circuit, N)

    for i in range(N):
        circuit.append("DEPOLARIZE1", N+i, p_prep)
    # TICK 4
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit  = stim.Circuit()
    for i in range(N):
        circuit.append("CNOT", [i, i+N])
        tick_circuit.append("CNOT", [i, i+N])
        circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)
    # TICK 5
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("X_ERROR", i+N, p_meas)
    # TICK 6
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MZ", i+N)
        tick_circuit.add_MZ(i+N)
    circuit += Z_stab_detector_circuit
    # TICK 7
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    ################## SWAPs ####################
    tick_circuit = stim.Circuit()
    if perm is not None:
        swaps1, swaps2 = decompose_into_two_involutions(perm)
        for (i,j) in swaps1:
            circuit.append("SWAP", [i, j])
            tick_circuit.append("SWAP", [i, j])
        for (i,j) in swaps2:
            circuit.append("SWAP", [i, j])
            tick_circuit.append("SWAP", [i, j])
    # TICK 8
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")
    #############################################

    tick_circuit = stim.Circuit()
    # the CNOT
    for i in range(N):
        circuit.append("CNOT", [i, i+2*N])
        tick_circuit.append("CNOT", [i, i+2*N])
        circuit.append("DEPOLARIZE2", [i, i+2*N], p_CNOT)
    # TICK 9
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
    # TICK 10
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
    # TICK 11
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("Z_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("Z_ERROR", i+3*N, p_meas)
    # TICK 12
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
    # TICK 13
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
    # TICK 14
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
    # TICK 15
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("X_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("X_ERROR", i+3*N, p_meas)
    # TICK 16
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
    # TICK 17
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")


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
            pauli_string_before_CNOT = propagate(form_pauli_string(rep_loc.flipped_pauli_product, 4*N), tick_circuits[tick:8])
            pauli_string_control_before_CNOT = pauli_string_before_CNOT[:N]
            pauli_string_target_before_CNOT = pauli_string_before_CNOT[2*N:3*N]
            # print(f"fault at tick={tick}, just before the CNOT, on control {format_pauli_string(pauli_string_control_before_CNOT)}; on target {format_pauli_string(pauli_string_target_before_CNOT)}")
        if (np.array(tick_list) < 8).all(): # before the transversal CNOT
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


def get_plus_triple_zero_prep_setup():
    print(f"get_plus_triple_zero_prep_setup: p_CNOT={p_CNOT}, p_meas={p_meas}, p_reset={p_reset}, p_idle={p_idle}, p_prep={p_prep}")
    tick_circuits = [] # for PauliString.after
    circuit = stim.Circuit()
    # data patch 1, in 0000
    append_initialization(circuit, "0000", 0)
    append_hypercube_encoding_circuit(circuit, 0)
    # ancilla patch for data patch 1, first in 0000 for phase EC
    append_initialization(circuit, "0000", N)
    append_hypercube_encoding_circuit(circuit, N)
    # data patch 2, in ++++
    append_initialization(circuit, "++++", 2*N)
    append_hypercube_encoding_circuit(circuit, 2*N)
    # ancilla patch for data patch 2, first in 0000 for phase EC
    append_initialization(circuit, "0000", 3*N)
    append_hypercube_encoding_circuit(circuit, 3*N)

    # state prep error on all patches
    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_prep)
        circuit.append("DEPOLARIZE1", N+i, p_prep)
        circuit.append("DEPOLARIZE1", 2*N+i, p_prep)
        circuit.append("DEPOLARIZE1", 3*N+i, p_prep)
    # TICK 0
    circuit.append("TICK")

    # first control-permuted CNOT
    tick_circuit  = stim.Circuit()
    for i in range(N):
        new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
        new_i = N-1-bin2int(new_i_vec)
        circuit.append("CNOT", [new_i+2*N, i])
        tick_circuit.append("CNOT", [new_i+2*N, i])
        circuit.append("DEPOLARIZE2", [new_i+2*N, i], p_CNOT)
    # TICK 1
    tick_circuits.append(tick_circuit)
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
    # TICK 2
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("Z_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("Z_ERROR", i+3*N, p_meas)
    # TICK 3
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
    # TICK 4
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
    # TICK 5
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
    # TICK 6
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("X_ERROR", i+N, p_meas)
        circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        circuit.append("X_ERROR", i+3*N, p_meas)
    # TICK 7
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
    # TICK 8
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # second control-permuted CNOT
    tick_circuit = stim.Circuit()
    for i in range(N):
        new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
        new_i = N-1-bin2int(new_i_vec)
        circuit.append("CNOT", [new_i+2*N, i])
        tick_circuit.append("CNOT", [new_i+2*N, i])
        circuit.append("DEPOLARIZE2", [new_i+2*N, i], p_CNOT)
    # TICK 9
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # Trailing Steane EC only on the top data block
    # immediate MX on the second data block
    # first, phase EC
    # re-initialize the ancilla
    append_initialization(circuit, "0000", N)
    append_hypercube_encoding_circuit(circuit, N)

    for i in range(N):
        circuit.append("DEPOLARIZE1", N+i, p_prep)
        circuit.append("Z_ERROR", 2*N+i, p_meas)
    # TICK 10
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = stim.Circuit()
    for i in range(N):
        circuit.append("CNOT", [i+N, i])
        tick_circuit.append("CNOT", [i+N, i])
        circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)
    # TICK 11
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MX", i+2*N)
        tick_circuit.add_MX(i+2*N)
    circuit += X_stab_detector_circuit
    circuit += X_log_detector_circuit # it is okay to add logical 1,2,3 because they are deterministically zero, while logical 0 is not.
    # TICK 12
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("Z_ERROR", i+N, p_meas)
    # TICK 13
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MX", i+N)
        tick_circuit.add_MX(i+N)
    circuit += X_stab_detector_circuit
    # TICK 14
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    # next, bit EC
    # re-initialize the ancilla
    append_initialization(circuit, "++++", N)
    append_hypercube_encoding_circuit(circuit, N)

    for i in range(N):
        circuit.append("DEPOLARIZE1", N+i, p_prep)
    # TICK 15
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = stim.Circuit()
    for i in range(N):
        circuit.append("CNOT", [i, i+N])
        tick_circuit.append("CNOT", [i, i+N])
        circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)
    # TICK 16
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    for i in range(N):
        circuit.append("DEPOLARIZE1", i, p_idle)
        circuit.append("X_ERROR", i+N, p_meas)
    # TICK 17
    tick_circuits.append(stim.Circuit())
    circuit.append("TICK")

    tick_circuit = MeasurementCircuit()
    for i in range(N):
        circuit.append("MZ", i+N)
        tick_circuit.add_MZ(i+N)
    circuit += Z_stab_detector_circuit
    # TICK 18
    tick_circuits.append(tick_circuit)
    circuit.append("TICK")

    diagram = circuit.diagram('timeline-svg')
    with open(f'GHZ_first_block_prep.svg', 'w') as f:
        print(diagram, file=f)

    dem = circuit.detector_error_model()
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
    dem_check_matrix = np.zeros((num_detectors, num_faults), dtype=np.uint8) # type is important, if type were bool, sum gives unexpected behaviour (OR instead of XOR)
    obs_matrix = np.zeros((K, num_faults), dtype=np.uint8)
    priors = np.zeros((num_faults), dtype=np.double)
    prop_dict = {}
    key_set = set()
    fault_explanation_dict = {}
    for i in range(len(flat_error_instructions)):

        dets = []
        frames = []
        instruction = flat_error_instructions[i]
        priors[i] = instruction.args_copy()[0]
        for t in instruction.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                frames.append(t.val)
        key = " ".join([f"D{s}" for s in sorted(dets)])
        # print(f"instruction i={i} probability={priors[i]} triggers detectors {key}")
        key_set.add(key)
        for det_id in dets:
            dem_check_matrix[det_id, i] = 1

        dem_filter = stim.DetectorErrorModel()
        dem_filter.append(flat_error_instructions[i])
        explained_errors: List[stim.ExplainedError] = circuit.explain_detector_error_model_errors(dem_filter=dem_filter, reduce_to_one_representative_error=False)
        tick_list = []
        final_pauli_string_all = []
        fault_explanation_dict[i] = explained_errors[0].circuit_error_locations[0]
        for rep_loc in explained_errors[0].circuit_error_locations:
            # print(rep_loc)
            tick = rep_loc.tick_offset
            tick_list.append(tick)
            final_pauli_string_before_MX = propagate(form_pauli_string(rep_loc.flipped_pauli_product, 4*N), tick_circuits[tick:11]) # just before the MX on the second data patch
            final_pauli_string_MX = final_pauli_string_before_MX[2*N:3*N]
            # sanity checks: if self calculation by propagation coincides with stim
            obs = Lx @ final_pauli_string_MX.to_numpy()[1].T % 2
            obs_matrix[:,i] = obs
            obs_nnz = list(np.nonzero(obs[1:])[0])
            assert obs_nnz == frames
            stab = low_wt_Hx @ final_pauli_string_MX.to_numpy()[1].T % 2
            stab_nnz = list(np.nonzero(stab)[0])
            stim_stab_nnz = [Did-120 for Did in dets if (Did>=120 and Did<(120+32))]
            assert stab_nnz == stim_stab_nnz

            final_pauli_string = propagate(form_pauli_string(rep_loc.flipped_pauli_product, 4*N), tick_circuits[tick:])
            final_pauli_string = final_pauli_string[:N]
            final_pauli_string_all.append(final_pauli_string)
            # if final_pauli_string.weight > 1:
            #     print(f"fault at tick={tick}, {format_pauli_string(final_pauli_string)}")

        if (np.array(tick_list) <= 10).all(): # before (and include) the second control-permuted CNOT, also including the Z_ERROR before the MX
            # check residual Pauli string all equal
            assert pauli_string_is_all_equal(final_pauli_string_all)
            # in sliding window decoding, only commit to those faults
            prop_dict[i] = final_pauli_string
        elif len([Did for Did in dets if (Did>=120 and Did<(120+32))]) > 0:
            print(explained_errors[0].circuit_error_locations)
            assert pauli_string_is_all_equal(final_pauli_string_all)
        else:
            assert not obs.any()
    
    end = time.time()
    print(f"Total time for constructing DEM and prop dict: {end-start}")   
    print("unique key (triggered detector string)", len(key_set))
    print("faults in prop_dict (commit to in sliding window)", len(prop_dict.keys()))
    print("rank of dem_check_matrix", rank(dem_check_matrix))
    dem_check_matrix_indices = [np.nonzero(row)[0] for row in dem_check_matrix]
    return dem_check_matrix_indices, obs_matrix, priors, prop_dict

def simulate_plus_triple_zero_prep_three_flag(num_shots, index):

    print(f"simulate_plus_triple_zero_prep_three_flag: p_CNOT={p_CNOT}, p_meas={p_meas}, p_reset={p_reset}, p_idle={p_idle}, p_prep={p_prep}")
    # prepare |+000>
    loader = AncillaErrorLoader()
    data_zero = loader.sample_ancilla_error(num_shots, index, f"logs_prep_zero/{dir_error_rate}")
    data_plus = loader.sample_ancilla_error(num_shots, index, f"logs_prep_plus/{dir_error_rate}")
    data_zero = loader.process_ancilla_error(data_zero, 'zero')
    data_plus = loader.process_ancilla_error(data_plus, 'plus')
    ancilla_top_zero = loader.sample_ancilla_error(num_shots, index+1, f"logs_prep_zero/{dir_error_rate}")
    ancilla_top_plus = loader.sample_ancilla_error(num_shots, index+1, f"logs_prep_plus/{dir_error_rate}")
    ancilla_top_zero = loader.process_ancilla_error(ancilla_top_zero, 'zero')
    ancilla_top_plus = loader.process_ancilla_error(ancilla_top_plus, 'plus')
    ancilla_bottom_zero = loader.sample_ancilla_error(num_shots, index+2, f"logs_prep_zero/{dir_error_rate}")
    ancilla_bottom_plus = loader.sample_ancilla_error(num_shots, index+2, f"logs_prep_plus/{dir_error_rate}")
    ancilla_bottom_zero = loader.process_ancilla_error(ancilla_bottom_zero, 'zero')
    ancilla_bottom_plus = loader.process_ancilla_error(ancilla_bottom_plus, 'plus')
    ancilla_top_zero_TEC = loader.sample_ancilla_error(num_shots, index+3, f"logs_prep_zero/{dir_error_rate}")
    ancilla_top_plus_TEC = loader.sample_ancilla_error(num_shots, index+3, f"logs_prep_plus/{dir_error_rate}")
    ancilla_top_zero_TEC = loader.process_ancilla_error(ancilla_top_zero_TEC, 'zero')
    ancilla_top_plus_TEC = loader.process_ancilla_error(ancilla_top_plus_TEC, 'plus')


    sim = stim.FlipSimulator(batch_size=num_shots, num_qubits=4*N)

    circuit = stim.Circuit()
    # data patch 1, in 0000
    append_initialization(circuit, "0000", 0)
    append_hypercube_encoding_circuit(circuit, 0)
    # ancilla patch for data patch 1, first in 0000 for phase EC
    append_initialization(circuit, "0000", N)
    append_hypercube_encoding_circuit(circuit, N)
    # data patch 2, in ++++
    append_initialization(circuit, "++++", 2*N)
    append_hypercube_encoding_circuit(circuit, 2*N)
    # ancilla patch for data patch 2, first in 0000 for phase EC
    append_initialization(circuit, "0000", 3*N)
    append_hypercube_encoding_circuit(circuit, 3*N)
    sim.do(circuit)
    # inject preparation noise
    prep_noise = [e1+e2+e3+e4 for (e1,e2,e3,e4) in zip(data_zero, ancilla_top_zero, data_plus, ancilla_bottom_zero)]
    X_component, Z_component = np.array([e.to_numpy() for e in prep_noise]).transpose(1,2,0) # each shape (4*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    permuted_CNOT1 = stim.Circuit()
    for i in range(N):
        new_i_vec = perm1 @ np.array(int2bin(N-1-i)) % 2
        new_i = N-1-bin2int(new_i_vec)
        permuted_CNOT1.append("CNOT", [new_i+2*N, i])
        permuted_CNOT1.append("DEPOLARIZE2", [new_i+2*N, i], p_CNOT)
    sim.do(permuted_CNOT1)

    phase_EC_circuit = stim.Circuit()
    for i in range(N):
        phase_EC_circuit.append("CNOT", [i+N, i])
        phase_EC_circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)
        phase_EC_circuit.append("CNOT", [i+3*N, i+2*N])
        phase_EC_circuit.append("DEPOLARIZE2", [i+3*N, i+2*N], p_CNOT)
    for i in range(N):
        phase_EC_circuit.append("DEPOLARIZE1", i, p_idle)
        phase_EC_circuit.append("Z_ERROR", i+N, p_meas)
        phase_EC_circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        phase_EC_circuit.append("Z_ERROR", i+3*N, p_meas)
    for i in range(N):
        phase_EC_circuit.append("MX", i+N)
    for i in range(N):
        phase_EC_circuit.append("MX", i+3*N)
    sim.do(phase_EC_circuit)

    bit_EC_prep_circuit = stim.Circuit()
    append_initialization(bit_EC_prep_circuit, "++++", N)
    append_hypercube_encoding_circuit(bit_EC_prep_circuit, N)
    append_initialization(bit_EC_prep_circuit, "++++", 3*N)
    append_hypercube_encoding_circuit(bit_EC_prep_circuit, 3*N)
    sim.do(bit_EC_prep_circuit)

    # inject preparation noise for bit EC
    prep_noise = [e1+e2+e3+e4 for (e1,e2,e3,e4) in zip([stim.PauliString(N) for _ in range(num_shots)], ancilla_top_plus, 
                                                       [stim.PauliString(N) for _ in range(num_shots)], ancilla_bottom_plus)]
    X_component, Z_component = np.array([e.to_numpy() for e in prep_noise]).transpose(1,2,0) # each shape (4*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    bit_EC_circuit = stim.Circuit()
    for i in range(N):
        bit_EC_circuit.append("CNOT", [i, i+N])
        bit_EC_circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)
        bit_EC_circuit.append("CNOT", [i+2*N, i+3*N])
        bit_EC_circuit.append("DEPOLARIZE2", [i+2*N, i+3*N], p_CNOT)
    for i in range(N):
        bit_EC_circuit.append("DEPOLARIZE1", i, p_idle)
        bit_EC_circuit.append("X_ERROR", i+N, p_meas)
        bit_EC_circuit.append("DEPOLARIZE1", i+2*N, p_idle)
        bit_EC_circuit.append("X_ERROR", i+3*N, p_meas)
    for i in range(N):
        bit_EC_circuit.append("MZ", i+N)
    for i in range(N):
        bit_EC_circuit.append("MZ", i+3*N)
    sim.do(bit_EC_circuit)

    permuted_CNOT2 = stim.Circuit()
    for i in range(N):
        new_i_vec = perm2 @ np.array(int2bin(N-1-i)) % 2
        new_i = N-1-bin2int(new_i_vec)
        permuted_CNOT2.append("CNOT", [new_i+2*N, i])
        permuted_CNOT2.append("DEPOLARIZE2", [new_i+2*N, i], p_CNOT)
    for i in range(N):
        permuted_CNOT2.append("Z_ERROR", i+2*N, p_meas)
        permuted_CNOT2.append("MX", i+2*N)
        permuted_CNOT2.append("DEPOLARIZE1", i, p_idle)
    sim.do(permuted_CNOT2)

    phase_EC_prep_circuit = stim.Circuit()
    append_initialization(phase_EC_prep_circuit, "0000", N)
    append_hypercube_encoding_circuit(phase_EC_prep_circuit, N)
    sim.do(phase_EC_prep_circuit)

    # inject preparation noise for phase EC
    prep_noise = [e1+e2+e3+e4 for (e1,e2,e3,e4) in zip([stim.PauliString(N) for _ in range(num_shots)], ancilla_top_zero_TEC, 
                                                       [stim.PauliString(N) for _ in range(num_shots)], [stim.PauliString(N) for _ in range(num_shots)])]
    X_component, Z_component = np.array([e.to_numpy() for e in prep_noise]).transpose(1,2,0) # each shape (4*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    phase_EC_circuit = stim.Circuit()
    for i in range(N):
        phase_EC_circuit.append("CNOT", [i+N, i])
        phase_EC_circuit.append("DEPOLARIZE2", [i+N, i], p_CNOT)
    for i in range(N):
        phase_EC_circuit.append("DEPOLARIZE1", i, p_idle)
        phase_EC_circuit.append("Z_ERROR", i+N, p_meas)
    for i in range(N):
        phase_EC_circuit.append("MX", i+N)
    sim.do(phase_EC_circuit)

    bit_EC_prep_circuit = stim.Circuit()
    append_initialization(bit_EC_prep_circuit, "++++", N)
    append_hypercube_encoding_circuit(bit_EC_prep_circuit, N)
    sim.do(bit_EC_prep_circuit)

    # inject preparation noise for bit EC
    prep_noise = [e1+e2+e3+e4 for (e1,e2,e3,e4) in zip([stim.PauliString(N) for _ in range(num_shots)], ancilla_top_plus_TEC, 
                                                       [stim.PauliString(N) for _ in range(num_shots)], [stim.PauliString(N) for _ in range(num_shots)])]
    X_component, Z_component = np.array([e.to_numpy() for e in prep_noise]).transpose(1,2,0) # each shape (4*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    bit_EC_circuit = stim.Circuit()
    for i in range(N):
        bit_EC_circuit.append("CNOT", [i, i+N])
        bit_EC_circuit.append("DEPOLARIZE2", [i, i+N], p_CNOT)
    for i in range(N):
        bit_EC_circuit.append("DEPOLARIZE1", i, p_idle)
        bit_EC_circuit.append("X_ERROR", i+N, p_meas)
    for i in range(N):
        bit_EC_circuit.append("MZ", i+N)
    sim.do(bit_EC_circuit)

    # sampling
    residual_errors = sim.peek_pauli_flips()
    print(f"residual_errors shape ({len(residual_errors)},{len(residual_errors[0])})")
    measurement_result = sim.get_measurement_flips().T
    measurement_result = measurement_result.astype(np.uint8)
    print("measurement result shape", measurement_result.shape)

    dem_check_matrix_indices, obs_matrix, priors, prop_dict = get_plus_triple_zero_prep_setup()

    def commit_to_faults_and_get_correction(fault_list):
        # fault_list contains a list of fault indices that happened according to the decoder
        # use prop_dict to combine them
        final_pauli_string_all = []
        for index in fault_list:
            if index in prop_dict.keys():
                e = prop_dict[index]
                final_pauli_string_all.append(e)
        final_pauli_string = reduce(stim.PauliString.__mul__, final_pauli_string_all, stim.PauliString(N))
        print(f"fault_list: {fault_list}")
        print(f"combined fault {format_pauli_string(final_pauli_string)}")
        return final_pauli_string



    detector_syndrome = np.hstack((
        measurement_result[:,:N]      @ low_wt_Hx.T % 2,
        measurement_result[:,N:2*N]   @ low_wt_Hx.T % 2,
        measurement_result[:,2*N:3*N] @ low_wt_Hz.T % 2,
        measurement_result[:,3*N:4*N] @ low_wt_Hz.T % 2,
        measurement_result[:,4*N:5*N] @ low_wt_Hx.T % 2, # MX on the second data patch
        measurement_result[:,5*N:6*N] @ low_wt_Hx.T % 2,
        measurement_result[:,6*N:7*N] @ low_wt_Hz.T % 2
    ))
    print("detector_syndrome shape", detector_syndrome.shape)

    start = time.time()
    faults, _ = mle_decoder_gurobi_decode_using_check_matrix(
        check_matrix_error_indices=dem_check_matrix_indices,
        logical_matrix_error_indices=None,
        error_probabilities=priors,
        detector_shots=detector_syndrome,
        gurobi_n_threads=1
    )
    print(f"MLE decoding {num_shots} samples took {time.time()-start} seconds.")

    residual_errors_out_all = []
    postselect_mask = np.zeros(num_shots, dtype=np.bool_) # three flags
    second_patch_MX = measurement_result[:,4*N:5*N]
    noisy_Lx_0 = second_patch_MX @ Lx[0].T % 2
    noisy_Lx_rest = second_patch_MX @ Lx[1:].T % 2
    print(f"noisy_Lx_0 sum {noisy_Lx_0.astype(int).sum()} over {num_shots} shots")
    print(f"noisy_Lx_rest sum {noisy_Lx_rest.astype(int).sum()} over {num_shots} shots")
    for idx, fault_list in enumerate(faults):
        # print(f"measurement error on MX {np.nonzero(second_patch_MX[idx])[0]}")
        logical_measurement_result = obs_matrix[:,fault_list].sum(axis=1) % 2
        corr = commit_to_faults_and_get_correction(fault_list)
        residual_errors_out = residual_errors[idx][:N] * corr
        # print(f"idx {idx} residual_errors_out {format_pauli_string(residual_errors_out)}")
        postselect_mask[idx] = np.logical_xor(noisy_Lx_rest[idx], logical_measurement_result[1:]).any()
        if postselect_mask[idx]:
            print(f"idx {idx} need to post-select away")
        if logical_measurement_result[0] != noisy_Lx_0[idx]:
            # calculate Z logical corrections and inject to the residual error of the output patch
            inject_Z_logical = Lz[0].astype(np.bool_)
            inject_Z_logical = stim.PauliString.from_numpy(xs=np.zeros(N, dtype=np.bool_), zs=inject_Z_logical)
            residual_errors_out *= inject_Z_logical
            corr *= inject_Z_logical
            print(f"idx {idx} inject logical Z correction")
        residual_errors_out_all.append( residual_errors_out )

        # update measurement result
        xs, zs = corr.to_numpy()
        measurement_result[idx,5*N:6*N] = (zs + measurement_result[idx,5*N:6*N]) % 2
        measurement_result[idx,6*N:7*N] = (xs + measurement_result[idx,6*N:7*N]) % 2

    return residual_errors_out_all, postselect_mask, measurement_result[:,5*N:6*N], measurement_result[:,6*N:7*N]

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

def simulate_GHZ_preparation(num_shots, log_num_block, index=0):
    # construct noiseless circuit and count how many ancilla files are needed
    # add inverse circuit and verify the measurement result is all-zero 
    num_block = 2**log_num_block
    num_data_qubits = num_block * N

    # noiseless inverse circuit
    inverse_circuit = stim.Circuit()
    for r in range(log_num_block)[::-1]:
        sep = 2 ** r
        for i in range(sep):
            for j in range(N):
                inverse_circuit.append("CNOT", [i*N + j, (i+sep)*N + j])
    for i in range(2**log_num_block):
        append_hypercube_encoding_circuit(inverse_circuit, offset=i*N)
    # undo the CNOT ladder on the first block
    inverse_circuit.append("CNOT", [logical_indices[0], logical_indices[2]])
    inverse_circuit.append("CNOT", [logical_indices[1], logical_indices[3]])
    inverse_circuit.append("CNOT", [logical_indices[0], logical_indices[1]])
    append_measurement(inverse_circuit, "+000", offset=0, detector=True)
    for i in range(1, 2**log_num_block):
        append_measurement(inverse_circuit, "0000", offset=i*N, detector=True)

    # noisy forward circuit
    circuit = stim.Circuit()
    append_initialization(circuit, "+000", offset=0, permute=perm_CNOT_ladder)
    append_hypercube_encoding_circuit(circuit, offset=0, permute=perm_CNOT_ladder)
    for i in range(1, num_block):
        append_initialization(circuit, "0000", offset=i*N)
        append_hypercube_encoding_circuit(circuit, offset=i*N)

    start = time.time()
    a1, postselect_mask, X_meas_result, Z_meas_result = simulate_plus_triple_zero_prep_three_flag(num_shots, index)
    Steane_cnt = index + 4
    print(f"|+000> decoding time {time.time()-start} seconds.")

    # permute errors (permutation corresponds to the CNOT ladder in the first block)
    permute_a1 = []
    for err in a1:
        x, z = err.to_numpy()
        new_x, new_z = x.copy(), z.copy()
        new_x[perm_CNOT_ladder] = x
        new_z[perm_CNOT_ladder] = z
        permute_a1.append( stim.PauliString.from_numpy(xs=new_x, zs=new_z) )

    residual_errors_snapshot = [permute_a1] # a list of lists
    # TODO: if only one block, skip the rest

    # load all-zero's
    for i in range(1, num_block):
        loader = AncillaErrorLoader()
        a_zero = loader.sample_ancilla_error(num_shots, index=Steane_cnt, parent_dir=f"logs_prep_zero/{dir_error_rate}")
        residual_errors_snapshot.append( loader.process_ancilla_error(a_zero, 'zero') )
        Steane_cnt += 1

    # start simulation and inject noise
    sim = stim.FlipSimulator(batch_size=num_shots, num_qubits=2*num_data_qubits)
    sim.do(circuit) # initialization only
    # num_qubits will be expanded to num_block * N * 2 later
    data_residual_errors = []
    for i in range(num_shots):
        e = stim.PauliString()
        for block in residual_errors_snapshot:
            e += block[i]
        data_residual_errors.append(e)
    X_component, Z_component = np.array([e.to_numpy() for e in data_residual_errors]).transpose(1,2,0) # each shape (2*num_block*N, num_shots)
    sim.broadcast_pauli_errors(pauli='X', mask=X_component)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

    # CNOT circuit on the block level, used to propagate corrections
    logical_circuit_block_level = []
    for r in range(log_num_block):
        transversal_CNOT_circuit_block_level = stim.Circuit()
        sep = 2 ** r
        for i in range(sep):
            transversal_CNOT_circuit_block_level.append("CNOT", [i, i+sep])
        logical_circuit_block_level.append(transversal_CNOT_circuit_block_level)
            
    tick_id_to_offset = {}
    # helper function for adding Steane EC after a layer of CNOT
    def add_Steane_at_tick(sim, tick, Steane_cnt, num_measurement):
        num_ancilla_block = 2**tick
        print(f"add_Steane_at_tick, tick={tick}, num_ancilla_block={num_ancilla_block}")

        zero_prep_circuit = stim.Circuit()
        for i in range(num_ancilla_block):
            append_initialization(zero_prep_circuit, "0000", (i+num_block)*N)
            append_hypercube_encoding_circuit(zero_prep_circuit, (i+num_block)*N)

        sim.do(zero_prep_circuit)

        # inject residual errors on |0000> and |++++> patches
        loader = AncillaErrorLoader()
        ancilla_zero_block_residual_errors = []
        ancilla_plus_block_residual_errors = []
        for i in range(num_ancilla_block):
            a_zero = loader.sample_ancilla_error(num_shots, Steane_cnt, f"logs_prep_zero/{dir_error_rate}")
            a_plus = loader.sample_ancilla_error(num_shots, Steane_cnt, f"logs_prep_plus/{dir_error_rate}")
            ancilla_zero_block_residual_errors.append( loader.process_ancilla_error(a_zero, 'zero') )
            ancilla_plus_block_residual_errors.append( loader.process_ancilla_error(a_plus, 'plus') )
            Steane_cnt += 1

        data_ancilla_zero_residual_errors = []
        for i in range(num_shots):
            e = stim.PauliString(num_data_qubits) # no error to inject on data blocks
            for block in ancilla_zero_block_residual_errors:
                e += block[i]
            data_ancilla_zero_residual_errors.append(e)

        data_ancilla_plus_residual_errors = []
        for i in range(num_shots):
            e = stim.PauliString(num_data_qubits) # no error to inject on data blocks
            for block in ancilla_plus_block_residual_errors:
                e += block[i]
            data_ancilla_plus_residual_errors.append(e)

        X_component, Z_component = np.array([e.to_numpy() for e in data_ancilla_zero_residual_errors]).transpose(1,2,0)
        sim.broadcast_pauli_errors(pauli='X', mask=X_component)
        sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)

        zero_coupling_circuit = stim.Circuit()
        for i in range(N * num_ancilla_block):
            zero_coupling_circuit.append("CNOT", [i+num_data_qubits, i])
            zero_coupling_circuit.append("DEPOLARIZE2", [i+num_data_qubits, i], p_CNOT)

        for i in range(N * num_ancilla_block):
            zero_coupling_circuit.append("Z_ERROR", i+num_data_qubits, p_meas)
            zero_coupling_circuit.append("MX", i+num_data_qubits)

        sim.do(zero_coupling_circuit)

        plus_prep_circuit = stim.Circuit()
        for i in range(num_ancilla_block):
            append_initialization(plus_prep_circuit, "++++", (i+num_block)*N)
            append_hypercube_encoding_circuit(plus_prep_circuit, (i+num_block)*N)

        sim.do(plus_prep_circuit)

        X_component, Z_component = np.array([e.to_numpy() for e in data_ancilla_plus_residual_errors]).transpose(1,2,0)
        sim.broadcast_pauli_errors(pauli='X', mask=X_component)
        sim.broadcast_pauli_errors(pauli='Z', mask=Z_component)
        
        plus_coupling_circuit = stim.Circuit()
        for i in range(N * num_ancilla_block):
            plus_coupling_circuit.append("CNOT", [i, i+num_data_qubits])
            plus_coupling_circuit.append("DEPOLARIZE2", [i, i+num_data_qubits], p_CNOT)

        for i in range(N * num_ancilla_block):
            plus_coupling_circuit.append("X_ERROR", i+num_data_qubits, p_meas)
            plus_coupling_circuit.append("MZ", i+num_data_qubits)

        sim.do(plus_coupling_circuit)

        # map (tick, block_idx) to offset for retrieving measurement results
        for i in range(num_ancilla_block):
            tick_id_to_offset[(tick-1, i)] = (num_measurement+i, num_measurement+num_ancilla_block+i)
        num_measurement += 2 * num_ancilla_block
        return Steane_cnt, num_measurement


    # last layer of CNOT, add noiseless Steane EC
    def add_noiseless_Steane(sim, tick, num_measurement):
        num_ancilla_block = 2**tick
        print(f"add_noiseless_Steane, num_ancilla_block={num_ancilla_block}")

        zero_prep_circuit = stim.Circuit()
        for i in range(num_ancilla_block):
            append_initialization(zero_prep_circuit, "0000", (i+num_block)*N)
            append_hypercube_encoding_circuit(zero_prep_circuit, (i+num_block)*N)

        sim.do(zero_prep_circuit)

        zero_coupling_circuit = stim.Circuit()
        for i in range(N * num_ancilla_block):
            zero_coupling_circuit.append("CNOT", [i+num_data_qubits, i])

        for i in range(N * num_ancilla_block):
            zero_coupling_circuit.append("MX", i+num_data_qubits)

        sim.do(zero_coupling_circuit)

        plus_prep_circuit = stim.Circuit()
        for i in range(num_ancilla_block):
            append_initialization(plus_prep_circuit, "++++", (i+num_block)*N)
            append_hypercube_encoding_circuit(plus_prep_circuit, (i+num_block)*N)

        sim.do(plus_prep_circuit)

        plus_coupling_circuit = stim.Circuit()
        for i in range(N * num_ancilla_block):
            plus_coupling_circuit.append("CNOT", [i, i+num_data_qubits])

        for i in range(N * num_ancilla_block):
            plus_coupling_circuit.append("MZ", i+num_data_qubits)

        sim.do(plus_coupling_circuit)

        # map (tick, block_idx) to offset for retrieving measurement results
        for i in range(num_ancilla_block):
            tick_id_to_offset[(tick, i)] = (num_measurement+i, num_measurement+num_ancilla_block+i)
        num_measurement += 2 * num_ancilla_block
        return num_measurement

    num_measurement = 0
    for r in range(log_num_block):
        transversal_CNOT_circuit = stim.Circuit()
        sep = 2 ** r
        for i in range(sep):
            for j in range(N):
                transversal_CNOT_circuit.append("CNOT", [i*N + j, (i+sep)*N + j])
                transversal_CNOT_circuit.append("DEPOLARIZE2", [i*N + j, (i+sep)*N + j], p_CNOT)
        sim.do(transversal_CNOT_circuit)
        # add Steane EC to block N,...2N-1
        Steane_cnt, num_measurement = add_Steane_at_tick(sim, r+1, Steane_cnt, num_measurement)

    num_measurement = add_noiseless_Steane(sim, log_num_block, num_measurement)

    print("tick_id_to_offset:", tick_id_to_offset)
    measurement_result = sim.get_measurement_flips().T
    measurement_result = measurement_result.astype(np.uint8)
    print("measurement_result shape", measurement_result.shape)
    print(f"self maintained num_measurement (multiply by N={N} should give the second dimension of measurement_result)", num_measurement)

    def bit_phase_get_result(tick, idx):
        phase_start, bit_start = tick_id_to_offset[(tick, idx)]
        bit_start *= N
        phase_start *= N
        return measurement_result[:, bit_start:bit_start+N], measurement_result[:, phase_start:phase_start+N]

    # final corrections
    final_corrections_X_component = np.zeros((num_shots, num_data_qubits), dtype=np.uint8)
    final_corrections_Z_component = np.zeros((num_shots, num_data_qubits), dtype=np.uint8)
    def update_result(tick, idx, corr, type):
        phase_start, bit_start = tick_id_to_offset[(tick, idx)]
        start = bit_start if type == 'bit' else phase_start
        start *= N
        measurement_result[:, start:start+N] = (measurement_result[:, start:start+N] + corr) % 2
        if tick == log_num_block-1:
            print(f"update final corrections with type {type} at block {idx}")
            phase_start, bit_start = tick_id_to_offset[(log_num_block, idx)]
            start = bit_start if type == 'bit' else phase_start
            start *= N
            measurement_result[:, start:start+N] = (measurement_result[:, start:start+N] + corr) % 2
            if type == 'bit':
                final_corrections_X_component[:, idx*N:(idx+1)*N] = (final_corrections_X_component[:, idx*N:(idx+1)*N] + corr) % 2
            else:
                final_corrections_Z_component[:, idx*N:(idx+1)*N] = (final_corrections_Z_component[:, idx*N:(idx+1)*N] + corr) % 2

    def propagate_and_update_X_correction(tick, idx, corr):
        e = lists_to_pauli_string([idx], [], num_block)
        for tick_offset, c in enumerate(logical_circuit_block_level[tick+1:]):
            temp_tick = tick + tick_offset + 1
            e = e.after(c)
            print(f"propagate X from tick={tick} block={idx}: tick_offset={tick_offset}, e={e}")
            x, _ = e.to_numpy()
            for temp_idx in np.nonzero(x)[0]:
                update_result(temp_tick, temp_idx, corr, type='bit')
    
    def propagate_and_update_Z_correction(tick, idx, corr):
        e = lists_to_pauli_string([], [idx], num_block)
        for tick_offset, c in enumerate(logical_circuit_block_level[tick+1:]):
            temp_tick = tick + tick_offset + 1
            e = e.after(c)
            print(f"propagate Z from tick={tick} block={idx}: tick_offset={tick_offset}, e={e}")
            _, z = e.to_numpy()
            for temp_idx in np.nonzero(z)[0]:
                update_result(temp_tick, temp_idx, corr, type='phase')
    
    def commit_to_faults_and_get_correction(fault_list, prop_dict):
        # fault_list contains a list of fault indices that happened according to the decoder
        # use prop_dict to combine them
        final_pauli_string_control_all = []
        final_pauli_string_target_all = []
        for index in fault_list:
            if index in prop_dict.keys():
                (c, t) = prop_dict[index]
                final_pauli_string_control_all.append(c)
                final_pauli_string_target_all.append(t)
                # print(f"commit to fault {index}")
        final_pauli_string_control = reduce(stim.PauliString.__mul__, final_pauli_string_control_all, stim.PauliString(N))
        final_pauli_string_target = reduce(stim.PauliString.__mul__, final_pauli_string_target_all, stim.PauliString(N))
        # print(f"combined fault, on control {format_pauli_string(final_pauli_string_control)}; on target {format_pauli_string(final_pauli_string_target)}")
        return final_pauli_string_control, final_pauli_string_target

    def mle_decoder_get_residual_error(dem_check_matrix_indices, priors, prop_dict, detector_syndrome):
        start = time.time()
        faults, _ = mle_decoder_gurobi_decode_using_check_matrix(
            check_matrix_error_indices=dem_check_matrix_indices,
            logical_matrix_error_indices=None,
            error_probabilities=priors,
            detector_shots=detector_syndrome, # (num_shots, num_detectors)
            gurobi_n_threads=1
        )
        print(f"MLE decoding {num_shots} samples took {time.time()-start} seconds.")
        ctrl_bit_corr_all = []
        ctrl_phase_corr_all = []
        targ_bit_corr_all = []
        targ_phase_corr_all = []
        for fault_list in faults:
            ctrl_corr, targ_corr = commit_to_faults_and_get_correction(fault_list, prop_dict)
            ctrl_bit_corr, ctrl_phase_corr = ctrl_corr.to_numpy()
            targ_bit_corr, targ_phase_corr = targ_corr.to_numpy()
            ctrl_bit_corr_all.append(ctrl_bit_corr)
            ctrl_phase_corr_all.append(ctrl_phase_corr)
            targ_bit_corr_all.append(targ_bit_corr)
            targ_phase_corr_all.append(targ_phase_corr)
        return np.array(ctrl_bit_corr_all), np.array(ctrl_phase_corr_all), np.array(targ_bit_corr_all), np.array(targ_phase_corr_all)

    if num_block > 1:
        perm_dem_check_matrix_indices, priors, prop_dict = get_GHZ_CNOT_window_setup(perm=perm_CNOT_ladder)
        ctrl_bit_TEC, ctrl_phase_TEC = bit_phase_get_result(0, 0)
        targ_bit_TEC, targ_phase_TEC = bit_phase_get_result(0, 1)

        detector_syndrome = np.hstack((
            X_meas_result  @ low_wt_Hx.T % 2,
            Z_meas_result  @ low_wt_Hz.T % 2,
            ctrl_phase_TEC @ low_wt_Hx.T % 2,
            targ_phase_TEC @ low_wt_Hx.T % 2,
            ctrl_bit_TEC   @ low_wt_Hz.T % 2,
            targ_bit_TEC   @ low_wt_Hz.T % 2
        ))

        ctrl_bit_corr, ctrl_phase_corr, targ_bit_corr, targ_phase_corr = mle_decoder_get_residual_error(
            perm_dem_check_matrix_indices, priors, prop_dict, detector_syndrome)
        # propagate the correction
        propagate_and_update_X_correction(-1, 0, ctrl_bit_corr)
        propagate_and_update_Z_correction(-1, 1, targ_phase_corr)
        propagate_and_update_X_correction(-1, 1, targ_bit_corr)
        propagate_and_update_Z_correction(-1, 0, ctrl_phase_corr)

    dem_check_matrix_indices, priors, prop_dict = get_GHZ_CNOT_window_setup()

    for r in range(1, log_num_block):
        sep = 2 ** r
        list_tup = [(i, i+sep) for i in range(sep)]
        tick = r - 1
        for (ctrl, targ) in list_tup:
            ctrl_bit_LEC, ctrl_phase_LEC = bit_phase_get_result(tick, ctrl)
            ctrl_bit_TEC, ctrl_phase_TEC = bit_phase_get_result(tick+1, ctrl)
            targ_bit_TEC, targ_phase_TEC = bit_phase_get_result(tick+1, targ)
            detector_syndrome = np.hstack((
                ctrl_phase_LEC @ low_wt_Hx.T % 2,
                ctrl_bit_LEC   @ low_wt_Hz.T % 2,
                ctrl_phase_TEC @ low_wt_Hx.T % 2,
                targ_phase_TEC @ low_wt_Hx.T % 2,
                ctrl_bit_TEC   @ low_wt_Hz.T % 2,
                targ_bit_TEC   @ low_wt_Hz.T % 2
            ))
            ctrl_bit_corr, ctrl_phase_corr, targ_bit_corr, targ_phase_corr = mle_decoder_get_residual_error(
                dem_check_matrix_indices, priors, prop_dict, detector_syndrome)
            # propagate the correction
            propagate_and_update_X_correction(tick, ctrl, ctrl_bit_corr)
            propagate_and_update_Z_correction(tick, targ, targ_phase_corr)
            propagate_and_update_X_correction(tick, targ, targ_bit_corr)
            propagate_and_update_Z_correction(tick, ctrl, ctrl_phase_corr)

    noiseless_round_X_component = []
    noiseless_round_Z_component = []
    # look at the noiseless round
    for idx in range(num_block):
        bit_result, phase_result = bit_phase_get_result(log_num_block, idx)
        bit_correction = bit_flip_EC(bit_result)
        phase_correction = phase_flip_EC(phase_result)
        noiseless_round_X_component.append(bit_correction)
        noiseless_round_Z_component.append(phase_correction)
    noiseless_round_X_component = np.hstack(noiseless_round_X_component)
    noiseless_round_Z_component = np.hstack(noiseless_round_Z_component)
    X_component = (final_corrections_X_component + noiseless_round_X_component) % 2
    Z_component = (final_corrections_Z_component + noiseless_round_Z_component) % 2

    # inject corrections to the sim
    sim.broadcast_pauli_errors(pauli='X', mask=X_component.astype(np.bool_).T)
    sim.broadcast_pauli_errors(pauli='Z', mask=Z_component.astype(np.bool_).T)

    sim.do(inverse_circuit) # do the inverse circuit
    measurement_results = sim.get_detector_flips().T
    print("without post-selection: measurement (detector) result shape", measurement_results.shape)
    print(f"#shots={num_shots}, #errors={measurement_results.any(axis=1).astype(int).sum()}")

    print("postselect_mask shape", postselect_mask.shape)
    # with open(f"logs_sim_final/nb{num_block}_{dir_error_rate}_index{index}_period{Steane_EC_period}.npy", "wb") as f:
    #     np.save(f, measurement_results)
    #     np.save(f, postselect_mask)

    # post-selection
    keep = np.where(postselect_mask==0)[0]
    measurement_results = measurement_results[keep]
    num_kept_shots = measurement_results.shape[0]
    print("post-selected: measurement (detector) result shape", measurement_results.shape)
    print(f"with post-selection: #shots={num_kept_shots}, #errors={measurement_results.any(axis=1).astype(int).sum()}")
    print(f"Total elasped time {time.time()-start} seconds.")
    print(measurement_results.astype(int).sum(axis=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Simulate CNOT ladder for phantom QRM")
    parser.add_argument("--index", type=int, default=0, help="index of the file")
    parser.add_argument("--num_shots", type=int, default=1024, help="number of batch")
    parser.add_argument("--p_CNOT", type=float, help="physical error rate of CNOT")
    parser.add_argument("-nb", "--num_block", type=int, default=1, help="number of blocks used in the CNOT ladder")
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
    print(f"p_CNOT={p_CNOT}, p_meas={p_meas}, p_reset={p_reset}, p_idle={p_idle}, #blocks={num_block}, index={args.index}, #shots={num_shots}")
    dir_error_rate = "p" + str(p_CNOT).split('.')[1]
    decoder = PyDecoder_polar_SCL(l, m) # arbitrary state decoder

    log_num_block = int(np.log2(num_block))
    assert num_block == 2**log_num_block, "number of block must be a power of two"
    start = time.time()
    simulate_GHZ_preparation(num_shots, log_num_block, index=args.index)
    print(f"GHZ simluation time {time.time()-start} seconds.")
    # get_plus_triple_zero_prep_setup()
    # simulate_plus_triple_zero_prep_three_flag(num_shots, 0)
