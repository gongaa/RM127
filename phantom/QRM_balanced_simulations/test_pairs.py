import stim
print(stim.__version__)
import numpy as np
import time
import operator
import itertools
import random
import pathlib
from functools import reduce
from PyDecoder_polar import PyDecoder_polar_SCL
import structured_test
from structured_test import get_pcm, get_plus_pcm, sample_plus_perm, sample_zero_perm, extract_6x6_binary_arrays
from utils_matrix import inverse
import settings
from settings import int2bin, bin2int, bin_wt, append_hypercube_encoding_circuit, append_initialization, append_measurement

m = structured_test.m
N = structured_test.N
l = structured_test.l
K = structured_test.K
d = structured_test.d
# phantom QRM parameter [[2^m, m-l+1, 2^{m-l} / 2^l]]
print(f"test_pair.py: m={m}, N={N}, K={K}, d={d}")
state = structured_test.state
flip_type = structured_test.flip_type

decoder = PyDecoder_polar_SCL(l, m, 0 if state == '0' else 1)

def test_faults(A1, A2, flip_type, state="0", num_faults=3, verbose=False, b1=None, b2=None, r1=False, r2=False):
    
    Ax = lambda A, i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
    Ax_b = lambda A, b, i: N-1-bin2int((A @ np.array(int2bin(N-1-i)) + b) % 2)
    a1_permute = [Ax(A1, i) for i in range(N)]
    a2_permute = [Ax(A2, i) for i in range(N)]
    if b1 is not None:
        a1_permute = [Ax_b(A1, b1, i) for i in range(N)]
    if b2 is not None:
        a2_permute = [Ax_b(A2, b2, i) for i in range(N)]

    if state == "0":
        a1_pcm, a1_error_explain_dict, a1_residual_error_dict = get_pcm(a1_permute, flip_type, reverse=r1)
        a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_pcm(a2_permute, flip_type, reverse=r2)
    else:
        a1_pcm, a1_error_explain_dict, a1_residual_error_dict = get_plus_pcm(a1_permute, flip_type, reverse=r1)
        a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_plus_pcm(a2_permute, flip_type, reverse=r2)

    def get_residual_error(err):
        if flip_type == 0: # X-flip
            num_flip = decoder.decode_X_flip(list(np.where(err)[0]))
        elif flip_type == 1: # Z-flip
            num_flip = decoder.decode_Z_flip(list(np.where(err)[0]))
        return num_flip

    a1_inv_dict = {}
    a1_num_col = a1_pcm.shape[1]
    for i in range(a1_pcm.shape[1]): 
        key = int(''.join(a1_pcm[:,i].astype('str')), 2)
        if key in a1_inv_dict.keys():
            print("two different faults trigger the same set of detectors")
        else:
            a1_inv_dict[key] = i

    a2_inv_dict = {}
    a2_num_col = a2_pcm.shape[1]
    for i in range(a2_pcm.shape[1]): 
        key = int(''.join(a2_pcm[:,i].astype('str')), 2)
        if key in a2_inv_dict.keys():
            print("two different faults trigger the same set of detectors")
        else:
            a2_inv_dict[key] = i

    # print("Ancilla 1 inverse dictionary length:", len(a1_inv_dict))
    # print("Ancilla 2 inverse dictionary length:", len(a2_inv_dict))
    
    # print("test one fault on ancilla 1, one fault on ancilla 2")
    violation_cnt = 0
    for i in range(a1_num_col):
        key = int(''.join(a1_pcm[:,i].astype('str')), 2)
        if key in a2_inv_dict.keys():
            j = a2_inv_dict[key]
            final_error = a1_residual_error_dict[i]
            if final_error.sum() > 2:
                # print("(1,1): final error weight", final_error.sum())
                if verbose: print("(1,1): final error weight", final_error.sum(), "; residue error weight", get_residual_error(final_error))
                # print("triggers stabilizers (key)", bin(key), "length", len(bin(key)[2:]))
                # print("explained faults:")
                # print("on ancilla 1,", a1_error_explain_dict[i], "final error at", np.where(a1_residual_error_dict[i])[0])
                # print("on ancilla 2,", a2_error_explain_dict[j], "final error at", np.where(a2_residual_error_dict[j])[0])
                violation_cnt += 1
                if not verbose: return False
    if verbose: print("(1, 1): violation cnt", violation_cnt)
    if num_faults == 2: return True
    # print("test two faults on ancilla 1, one fault on ancilla 2, and create a1 two fault dictionary")
    violation_cnt = 0
    a1_two_faults_dict = {}
    for i in range(a1_num_col):
        for j in range(i+1, a1_num_col):
            xor = (a1_pcm[:,i] + a1_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a2_inv_dict.keys():
                k = a2_inv_dict[key]
                final_error = a1_residual_error_dict[i] ^ a1_residual_error_dict[j]
                if final_error.sum() > 3:
                    # print("(2,1): final error weight", final_error.sum(), "; residue error weight", get_residual_error(final_error))
                    violation_cnt += 1
                    return False
                    # print("explained faults:")
                    # print("on ancilla 1,", a1_error_explain_dict[i], "final error at", np.where(a1_residual_error_dict[i])[0])
                    # print("on ancilla 1,", a1_error_explain_dict[j], "final error at", np.where(a1_residual_error_dict[j])[0])
                    # print("on ancilla 2,", a2_error_explain_dict[k], "final error at", np.where(a2_residual_error_dict[k])[0])
                    
            if key not in a1_two_faults_dict.keys():
                a1_two_faults_dict[key] = a1_residual_error_dict[i] ^ a1_residual_error_dict[j]
    if verbose: print("(2, 1): violation cnt", violation_cnt)

    # print("test one fault on ancilla 1, two faults on ancilla 2, and create a2 two fault dictionary")
    violation_cnt = 0
    a2_two_faults_dict = {}
    for i in range(a2_num_col):
        for j in range(i+1, a2_num_col):
            xor = (a2_pcm[:,i] + a2_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a1_inv_dict.keys():
                k = a1_inv_dict[key]
                final_error = a1_residual_error_dict[k]
                if final_error.sum() > 3:
                    # print("(1,2): final error weight", final_error.sum(), "; residue error weight", get_residual_error(final_error))
                    violation_cnt += 1
                    return False
                    # print("explained faults:")
                    # print("on ancilla 1,", a1_error_explain_dict[k], "final error at", np.where(a1_residual_error_dict[k])[0])
                    # print("on ancilla 2,", a1_error_explain_dict[i], "final error at", np.where(a2_residual_error_dict[i])[0])
                    # print("on ancilla 2,", a2_error_explain_dict[j], "final error at", np.where(a2_residual_error_dict[j])[0])
            if key not in a2_two_faults_dict.keys():
                a2_two_faults_dict[key] = a2_residual_error_dict[i] ^ a2_residual_error_dict[j]
    
    if verbose: print("(1, 2): violation cnt", violation_cnt)
    if num_faults == 3:  return True
    # print("Ancilla 1 two fault dictionary length:", len(a1_two_faults_dict))
    # print("Ancilla 2 two fault dictionary length:", len(a2_two_faults_dict))
    violation_cnt = 0
    # print("test two fault on ancilla 1, two faults on ancilla 2")
    for k1, v1 in a1_two_faults_dict.items():
        if k1 in a2_two_faults_dict.keys():
            if v1.sum() > 4 and get_residual_error(v1) > 4:
                # print("2 faults on A1, 2 faults on A2, final error weight", v1.sum(), "; residue error weight", get_residual_error(v1), flush=True)
                violation_cnt += 1
                return False
    if verbose: print("(2, 2): violation cnt", violation_cnt)           

    # print("test three faults on ancilla 1 and one fault on ancilla 2, and vice versa")
    violation_cnt = 0
    for i in range(a1_num_col):
        for j in range(a2_num_col):
            xor = (a1_pcm[:,i] + a2_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a1_two_faults_dict.keys(): # one fault on ancilla 2
                final_error = a2_residual_error_dict[j]
                if final_error.sum() > 4 and get_residual_error(final_error) > 4:
                    # print("3 faults on A1, 1 fault on A2, final error weight", final_error.sum(), "; residue error weight", get_residual_error(final_error))
                    violation_cnt += 1
                    return False
            if key in a2_two_faults_dict.keys(): # one fault on ancilla 1
                final_error = a1_residual_error_dict[i]
                if final_error.sum() > 4 and get_residual_error(final_error) > 4:
                    # print("3 faults on A2, 1 fault on A1, final error weight", final_error.sum(), "; residues error weight", get_residual_error(final_error))
                    violation_cnt += 1
                    return False
    if verbose: print("(1, 3), (3, 1): violation cnt", violation_cnt) 
    return True

############## search for permutations for |0000> ################
# assert state == '0', "set state to zero"
# A1 = np.eye(m, dtype=int)
# if state == '0':
#     for _ in range(200000):
#         A2 = sample_zero_perm()
#         pass_test = test_faults(A1, A2, flip_type, state, num_faults=2)
#         if pass_test:
#             print(A2)
# exit()

# A2_arrays = []
# for filename in list(pathlib.Path('.').glob('zero_state_X_flip_nf2_*.log')):
#     with open(filename, "r") as f:
#         text = f.read()
#         A2_arrays += extract_6x6_binary_arrays(text)

# num_ones = [a.sum() for a in A2_arrays]
# print(f"avg #ones in A2 array: {sum(num_ones)/len(num_ones)}, max {max(num_ones)}, min {min(num_ones)}")

# A3_arrays = []
# for filename in list(pathlib.Path('.').glob('zero_state_Z_flip*.log')):
#     with open(filename, "r") as f:
#         text = f.read()
#         A3_arrays += extract_6x6_binary_arrays(text)

# num_ones = [a.sum() for a in A3_arrays]
# print(f"avg #ones in A3 array: {sum(num_ones)/len(num_ones)}, max {max(num_ones)}, min {min(num_ones)}")

# print("length of A2 array", len(A2_arrays))
# print("length of A3 array", len(A3_arrays))

# num_ones = []
# A1 = np.eye(m, dtype=int)
# for _ in range(1000000):
#     A2 = random.choice(A2_arrays) # pass 12 X-flip, three faults
#     A3 = random.choice(A3_arrays) # pass 13 Z-flip, three faults
#     pass_test_23 = test_faults(A2, A3, 1, "0", num_faults=2) # 23 Z-flip
#     if not pass_test_23: continue
#     A4 = random.choice(A3_arrays) # pass 14 Z-flip, three faults
#     pass_test_34 = test_faults(A3, A4, 0, "0", num_faults=2) # 34 X-flip
#     if not pass_test_34: continue
#     pass_test_24 = test_faults(A2, A4, 1, "0", num_faults=2) # 24 Z-flip
#     if not pass_test_24: continue
#     print("A2, A3, A4")
#     print(A2)
#     print(A3)
#     print(A4, flush=True)
# exit()
############### search for permutations for |++++> #################
# A1 = np.eye(m, dtype=int)
# assert state == '+', "set state to plus"
# for _ in range(200000):
#     A2 = sample_plus_perm()
#     pass_test = test_faults(A1, A2, flip_type, state, num_faults=2)
#     if pass_test:
#         print(A2)
# exit()
# Thoughts: hypercube circuit = its reverse
# not just code isomorphism, but circuit identity as well

A2_arrays = []
for filename in list(pathlib.Path('.').glob('plus_state_Z_flip_nf2_*.log')):
    with open(filename, "r") as f:
        text = f.read()
        A2_arrays += extract_6x6_binary_arrays(text)

num_ones = [a.sum() for a in A2_arrays]
print(f"avg #ones in A2 array: {sum(num_ones)/len(num_ones)}, max {max(num_ones)}, min {min(num_ones)}")

A3_arrays = []
for filename in list(pathlib.Path('.').glob('plus_state_X_flip_nf2_*.log')):
    with open(filename, "r") as f:
        text = f.read()
        A3_arrays += extract_6x6_binary_arrays(text)

num_ones = [a.sum() for a in A3_arrays]
print(f"avg #ones in A3 array: {sum(num_ones)/len(num_ones)}, max {max(num_ones)}, min {min(num_ones)}")

print("length of A2 array", len(A2_arrays))
print("length of A3 array", len(A3_arrays))

A1 = np.eye(m, dtype=int)
for _ in range(200000):
    A2 = random.choice(A2_arrays) # pass 12 Z-flip, two faults
    A3 = random.choice(A3_arrays) # pass 13 X-flip, three faults
    pass_test_23 = test_faults(A2, A3, 0, "+", num_faults=2) # 23 X-flip
    if not pass_test_23: continue
    A4 = random.choice(A3_arrays) # pass 14 X-flip, three faults
    pass_test_34 = test_faults(A3, A4, 1, "+", num_faults=2) # 34 Z-flip
    if not pass_test_34: continue
    print("pass 34")
    pass_test_24 = test_faults(A2, A4, 0, "+", num_faults=2, verbose=True) # 24 X-flip
    if not pass_test_24: continue
    print("A2, A3, A4")
    print(A2)
    print(A3)
    print(A4, flush=True)
exit()

# A1 = np.eye(m, dtype=int)
# for _ in range(200000):
#     A4 = random.choice(A3_arrays) # pass 14 X-flip, three faults
#     pass_test_14 = test_faults(A1, A4, 0, "+", num_faults=2, r2=True) # 14 X-flip
#     if not pass_test_14: continue
#     A2 = random.choice(A2_arrays) # pass 12 Z-flip, two faults
#     A3 = random.choice(A3_arrays) # pass 13 X-flip, three faults
#     pass_test_23 = test_faults(A2, A3, 0, "+", num_faults=2) # 23 X-flip
#     if not pass_test_23: continue
#     pass_test_34 = test_faults(A3, A4, 1, "+", num_faults=2, r2=True) # 34 Z-flip
#     if not pass_test_34: continue
#     print("pass 34")
#     pass_test_24 = test_faults(A2, A4, 0, "+", num_faults=2, r2=True, verbose=True) # 24 X-flip
#     if not pass_test_24: continue
#     print("A2, A3, A4")
#     print(A2)
#     print(A3)
#     print(A4, flush=True)
# exit()



# A1 = np.eye(m, dtype=int)
# for _ in range(20000):
#     A2 = random.choice(A2_arrays) # pass 12 Z-flip, two faults
#     A3 = random.choice(A3_arrays) # pass 13 X-flip, three faults
#     pass_test_12 = False
#     for _ in range(10):
#         b2 = np.random.randint(2, size=m)
#         pass_test_12 = test_faults(A1, A2, 1, "+", num_faults=2, b2=b2) # 12 Z-flip
#         if pass_test_12: break
#     if not pass_test_12: continue 
#     pass_test_23 = test_faults(A2, A3, 0, "+", num_faults=2, b1=b2) # 23 X-flip
#     if not pass_test_23: continue
#     A4 = random.choice(A3_arrays) # pass 14 X-flip, three faults
#     pass_test_14 = False
#     for _ in range(10):
#         b4 = np.random.randint(2, size=m)
#         pass_test_14 = test_faults(A1, A4, 0, "+", num_faults=2, b2=b4) # 14 X-flip
#         if pass_test_14: break
#     if not pass_test_14: continue
#     pass_test_34 = test_faults(A3, A4, 1, "+", num_faults=2, b2=b4) # 34 Z-flip
#     if not pass_test_34: continue
#     print("pass 34")
#     pass_test_24 = test_faults(A2, A4, 0, "+", num_faults=2, b1=b2, b2=b4, verbose=True) # 24 X-flip
#     if not pass_test_24: continue
#     print("pass 24")
#     print("A2, A3, A4")
#     print(np.hstack((A2,b2)))
#     print(A3)
#     print(np.hstack((A4,b4)), flush=True)


