from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import time, argparse
import stim
from utils import dem_to_check_matrices
from pysat.examples.rc2 import RC2
from pysat.formula import WCNF
from typing import List, Tuple, Optional
import re
import settings
from settings import *
import sys
sys.path.append("../")
from src.simulation_v2.decoder_mle import *

# change m to 4, l to 2, and logical_indices_binary in line 114 in settings.py
m = settings.m
N = settings.N
l = settings.l
K = settings.K
d = settings.d

_TOKEN_RE = re.compile(r'(?:(?<=\s)|^)(CZ|S|Z)')   # token at start or after whitespace
_BRACKETS_RE = re.compile(r"\[([^\]]*)\]")        # content inside [...]
_INT_RE = re.compile(r"-?\d+")

def extract_groups_with_z(s: str) -> Tuple[List[Tuple[int, ...]], Optional[List[Tuple[int, ...]]]]:
    """
    Returns:
      main_groups: tuples from S and CZ sections
      z_groups: tuples from Z section, or None if no Z section/groups exist

    Robust to extra commas, e.g. Z[,1,,2] -> (1,2)
    """
    main: List[Tuple[int, ...]] = []
    z: List[Tuple[int, ...]] = []

    matches = list(_TOKEN_RE.finditer(s))
    if not matches:
        return [], None

    for i, m in enumerate(matches):
        token = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(s)
        section = s[start:end]

        for g in _BRACKETS_RE.findall(section):
            nums = _INT_RE.findall(g)  # ignore extra commas / junk, just take integers
            if not nums:
                continue
            tup = tuple(map(int, nums))

            if token == "Z":
                z.append(tup)
            else:  # S or CZ
                main.append(tup)

    return main, (z if z else None)

def extract_groups(s: str) -> List[Tuple[int, ...]]:
    """
    Extract bracketed integer groups from a string like:
    S[0][7][10][13] CZ[,1,11][2,9]...
    Extra commas are ignored. Empty groups are skipped.
    """
    groups = re.findall(r"\[([^\]]*)\]", s)  # content inside each [...]
    out: List[Tuple[int, ...]] = []

    for g in groups:
        nums = re.findall(r"-?\d+", g)        # all integers in the group
        if not nums:
            continue
        out.append(tuple(map(int, nums)))

    return out


def get_circuit_level_distance(physical_string, logical_string):
    p = 0.001
    logical_gate, Z_logical_gate = extract_groups_with_z(logical_string)
    involution = extract_groups(physical_string)
    logical_circuit = stim.Circuit()
    for tup in logical_gate:
        if len(tup) == 1:
            logical_circuit.append("S_DAG", logical_indices[tup[0]])
        elif len(tup) == 2:
            logical_circuit.append("CZ", [logical_indices[tup[0]], logical_indices[tup[1]]])
    if Z_logical_gate is not None:
        for tup in Z_logical_gate:
            logical_circuit.append("Z", logical_indices[tup[0]])
    circuit = stim.Circuit()
    append_initialization(circuit, state, offset=0)
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
    # undo the logical circuit
    circuit += logical_circuit

    append_measurement(circuit, state, offset=0)
    # append logical observable
    for i in range(K):
        obs_str = f"OBSERVABLE_INCLUDE({i}) rec[{-K+i}]\n"
        circuit += stim.Circuit(obs_str)

    # print(f"#detectors={circuit.num_detectors}, #obs={circuit.num_observables}")
    try:
        SAT_str = circuit.shortest_error_sat_problem()
        wcnf = WCNF(from_string=SAT_str)
        with RC2(wcnf) as rc2:
            rc2.compute()
            d = rc2.cost
            # print(f"distance {d} at physical gate {involution}, logical gate {logical_gate}")
            return d
    except ValueError:
        print(f"Unidentified: physical string: {physical_string}, logical string: {logical_string}")
        return -1

res = depth_one_t(low_wt_Hx, Lx, t=2, verbosity=0) # set t=2 to search for fold-S
print(f"length of result: {len(res)}")
state = '+++'
for gate in res:
    d = get_circuit_level_distance(gate['physical_string'], gate['logical_string'])
    if d > 2:
        print(gate)

