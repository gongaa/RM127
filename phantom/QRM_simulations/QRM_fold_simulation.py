from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import time, argparse
import stim
from utils import dem_to_check_matrices
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
import settings
from settings import *
import sys
sys.path.append("../")
from src.simulation_v2.decoder_mle import *

m = settings.m
N = settings.N
l = settings.l
K = settings.K
d = settings.d

########### fold-S ################
permutation = [i for i in range(N)]
for i in range(N):
    i_bin = int2bin(i)
    i_bin = [1-i_bin[5], 1-i_bin[4], 1-i_bin[3], 1-i_bin[2], 1-i_bin[1], 1-i_bin[0]] # SS
    # i_bin = [i_bin[5], i_bin[4], 1-i_bin[3], 1-i_bin[2], i_bin[1], i_bin[0]] # also SS
    # i_bin = [1-i_bin[5], 1-i_bin[4], 1-i_bin[2], 1-i_bin[3], 1-i_bin[1], 1-i_bin[0]] # CZ
    permutation[i] = bin2int(i_bin)
fixed_point = [i for (idx, i) in enumerate(permutation) if i==idx]
print("fixed points", fixed_point)
# need to check INVOLUTION !!!!!!!
involution = []
for i in range(N):
    j = permutation[i]
    if i != j:
        assert permutation[j] == i, "Not an involution!"
        temp = sorted([i,j])
        if temp not in involution:
            involution.append(temp)
    else:
        involution.append([i])

print(f"length of involution: {len(involution)}")
for tup in involution:
    if len(tup) == 1: continue
    for lx in Lx:
        lx_supp = list(np.nonzero(lx)[0])
        # print(f"tup={tup}, lx_supp={lx_supp}")
        if tup[0] in lx_supp and tup[1] in lx_supp:
            print(f"CZ between {tup[0]} and {tup[1]} both in a logical X support {lx_supp}")
    for lz in Lz:
        lz_supp = list(np.nonzero(lz)[0])
        if tup[0] in lz_supp and tup[1] in lz_supp:
            print(f"CZ between {tup[0]} and {tup[1]} both in a logical Z support {lz_supp}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Simulate fold-S for phantom QRM")
    parser.add_argument("--num_shots", type=int, default=1024, help="number of shots")
    parser.add_argument("--p", type=float, help="physical error rate of gates")
    args = parser.parse_args()
    p = args.p
    num_shots = args.num_shots
    circuit = stim.Circuit()
    append_initialization(circuit, "+000", offset=0)
    append_hypercube_encoding_circuit(circuit, offset=0)
    # for i in range(N):
    #     circuit.append("DEPOLARIZE1", i, p)
    for tup in involution:
        if len(tup) == 1:
            circuit.append("S", tup[0])
            circuit.append("DEPOLARIZE1", tup[0], p)
        else:
            circuit.append("CZ", [tup[0], tup[1]])
            circuit.append("DEPOLARIZE2", [tup[0], tup[1]], p)

    # noiseless stabilizer measurements via MPP
    for i, s in enumerate(low_wt_Hx):
        nnz = np.nonzero(s)[0]
        circuit.append("MPP", [*stim.target_combined_paulis([stim.target_x(nnz_idx) for nnz_idx in nnz])])
        circuit.append("DETECTOR", stim.target_rec(-1))

    for i, s in enumerate(low_wt_Hz):
        nnz = np.nonzero(s)[0]
        circuit.append("MPP", [*stim.target_combined_paulis([stim.target_z(nnz_idx) for nnz_idx in nnz])])
        circuit.append("DETECTOR", stim.target_rec(-1))

    # noiseless unencode
    append_hypercube_encoding_circuit(circuit, offset=0)
    # undo the logical SSII using S dagger's
    # SS: circuit-level distance 4 and 7 for state '++++' and '0000', distance 4 and 4 for state '++00' and '00++'
    circuit.append("S_DAG", logical_indices[0])
    circuit.append("S_DAG", logical_indices[1])
    # CZ: circuit-level distance 4 and 6 for state '++++' and '0000', distance 4 and 4 for state '++00' and '00++'
    # circuit.append("CZ", [logical_indices[0], logical_indices[1]])
    append_measurement(circuit, "+000", offset=0)
    # append logical observable
    for i in range(K):
        obs_str = f"OBSERVABLE_INCLUDE({i}) rec[{-K+i}]\n"
        circuit += stim.Circuit(obs_str)

    print(f"#detectors={circuit.num_detectors}, #obs={circuit.num_observables}")
    SAT_str = circuit.shortest_error_sat_problem()

    wcnf = WCNF(from_string=SAT_str)
    with RC2(wcnf) as rc2:
        rc2.compute()
        print("circuit level distance", rc2.cost)

    diagram = circuit.diagram('timeline-svg')
    with open(f'fold-S.svg', 'w') as f:
        print(diagram, file=f)

    dem = circuit.detector_error_model()
    chk, obs, priors, col_dict = dem_to_check_matrices(dem, return_col_dict=True)

    num_row, num_col = chk.shape
    chk_row_wt = np.sum(chk, axis=1)
    chk_col_wt = np.sum(chk, axis=0)
    print(f"check matrix shape {chk.shape}, max (row, column) weight ({np.max(chk_row_wt)}, {np.max(chk_col_wt)}),",
          f"min (row, column) weight ({np.min(chk_row_wt)}, {np.min(chk_col_wt)})")

    dem_sampler: stim.CompiledDemSampler = dem.compile_sampler()
    det_data, obs_data, err_data = dem_sampler.sample(shots=num_shots, return_errors=False, bit_packed=False)
    print(f"det_data shape {det_data.shape}", f"num nnz: {det_data.sum()}")

    start = time.time()
    _, decoded_logicals = mle_decoder_gurobi_decode_using_dem(dem, det_data)
    print(f"MLE decoding {num_shots} shots took {time.time()-start} seconds")
    print("decoded_logical shape", decoded_logicals.shape, "obs_data shape", obs_data.shape)
    # compare obs_data with decoded_logicals
    log_diff = np.logical_xor(obs_data, decoded_logicals)
    num_errors = log_diff.any(axis=1).astype(int).sum()
    print(f"number of logical errors: {num_errors} in {num_shots} shots")
    print(f"over the four logicals: {log_diff.astype(int).sum(axis=0)}")
    print(num_errors/num_shots)

