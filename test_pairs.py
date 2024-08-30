import stim
print(stim.__version__)
import numpy as np
import time
import operator
import itertools
import random
from functools import reduce
from structured_test import get_pcm, get_plus_pcm

n = 7 # keep it the same as in structure_test.py
N = 2 ** n 
d = 15
# also keep d and wt_thresh the same as in structure_test.py
if d == 7:
    wt_thresh = n - (n-1)//3 # for [[127,1,7]]
elif d == 15:
    wt_thresh = n - (n-1)//2 # for [[127,1,15]]
else:
    print("unsupported distance", d)

print(f"test_pair.py: n={n}, N={N}, d={d}, wt_thresh={wt_thresh}")

int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(n, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)

def Eij(i,j):
    A = np.eye(n, dtype=int)
    A[i,j] = 1
    return A

def ce(exclude, l=0, u=n): # choose except
    choices = set(range(l,u)) - set([exclude])
    return random.choice(list(choices))

def test_faults(A, flip_type, state="0"):
    
    a1_permute = [i for i in range(N-1)]
    a1_pcm, a1_error_explain_dict, a1_residual_error_dict = get_pcm(a1_permute, flip_type)
    a1_inv_dict = {}
    a1_num_col = a1_pcm.shape[1]
    for i in range(a1_pcm.shape[1]): 
        key = int(''.join(a1_pcm[:,i].astype('str')), 2)
        if key in a1_inv_dict.keys():
            print("two different faults trigger the same set of detectors")
        else:
            a1_inv_dict[key] = i

    Ax = lambda i: N-1-bin2int(A @ np.array(int2bin(N-1-i)) % 2)
    a2_permute = [Ax(i) for i in range(N-1)]

    if state == "0":
        a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_pcm(a2_permute, flip_type)
    else:
        a2_pcm, a2_error_explain_dict, a2_residual_error_dict = get_plus_pcm(a2_permute, flip_type)


    a2_inv_dict = {}
    a2_num_col = a2_pcm.shape[1]
    for i in range(a2_pcm.shape[1]): 
        key = int(''.join(a2_pcm[:,i].astype('str')), 2)
        if key in a2_inv_dict.keys():
            print("two different faults trigger the same set of detectors")
        else:
            a2_inv_dict[key] = i

#     print("Ancilla 1 inverse dictionary length:", len(a1_inv_dict))
#     print("Ancilla 2 inverse dictionary length:", len(a2_inv_dict))
    
#     print("test one fault on ancilla 1, one fault on ancilla 2")
    for i in range(a1_num_col):
        key = int(''.join(a1_pcm[:,i].astype('str')), 2)
        if key in a2_inv_dict.keys():
            j = a2_inv_dict[key]
            final_error = a1_residual_error_dict[i]
            if final_error.sum() > 2: #1:
                return False
                print("final error weight", final_error.sum())
                print("explained faults:")
                print("on ancilla 1,", a1_error_explain_dict[i], "final error at", np.where(a1_residual_error_dict[i])[0])
                print("on ancilla 2,", a2_error_explain_dict[j], "final error at", np.where(a2_residual_error_dict[j])[0])

#     print("test two faults on ancilla 1, one fault on ancilla 2, and create a1 two fault dictionary")
    a1_two_faults_dict = {}
    for i in range(a1_num_col):
        for j in range(i+1, a1_num_col):
            xor = (a1_pcm[:,i] + a1_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a2_inv_dict.keys():
                k = a2_inv_dict[key]
                final_error = a1_residual_error_dict[i] ^ a1_residual_error_dict[j]
                if final_error.sum() > 3: #2:
                    return False
                    print("final error weight", final_error.sum())
                    print("explained faults:")
                    print("on ancilla 1,", a1_error_explain_dict[i], "final error at", np.where(a1_residual_error_dict[i])[0])
                    print("on ancilla 1,", a1_error_explain_dict[j], "final error at", np.where(a1_residual_error_dict[j])[0])
                    print("on ancilla 2,", a2_error_explain_dict[k], "final error at", np.where(a2_residual_error_dict[k])[0])
                    
            if key not in a1_two_faults_dict.keys():
                a1_two_faults_dict[key] = a1_residual_error_dict[i] ^ a1_residual_error_dict[j]

#     print("test one fault on ancilla 1, two faults on ancilla 2, and create a2 two fault dictionary")
    a2_two_faults_dict = {}
    for i in range(a2_num_col):
        for j in range(i+1, a2_num_col):
            xor = (a2_pcm[:,i] + a2_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a1_inv_dict.keys():
                k = a1_inv_dict[key]
                final_error = a1_residual_error_dict[k]
                if final_error.sum() > 3: # 2:
                    return False
                    print("final error weight", final_error.sum())
                    print("explained faults:")
                    print("on ancilla 1,", a1_error_explain_dict[k], "final error at", np.where(a1_residual_error_dict[k])[0])
                    print("on ancilla 2,", a1_error_explain_dict[i], "final error at", np.where(a2_residual_error_dict[i])[0])
                    print("on ancilla 2,", a2_error_explain_dict[j], "final error at", np.where(a2_residual_error_dict[j])[0])
            if key not in a2_two_faults_dict.keys():
                a2_two_faults_dict[key] = a2_residual_error_dict[i] ^ a2_residual_error_dict[j]
    
#     print("Ancilla 1 two fault dictionary length:", len(a1_two_faults_dict))
#     print("Ancilla 2 two fault dictionary length:", len(a2_two_faults_dict))
#     print("test two fault on ancilla 1, two faults on ancilla 2")
    for k1, v1 in a1_two_faults_dict.items():
        if k1 in a2_two_faults_dict.keys():
            if v1.sum() > 4:
                return False
                print("final error weight", v1.sum())
                
#     print("test three faults on ancilla 1 and one fault on ancilla 2, and vice versa")
    for i in range(a1_num_col):
        for j in range(a2_num_col):
            xor = (a1_pcm[:,i] + a2_pcm[:,j]) % 2
            key = int(''.join(xor.astype('str')), 2)
            if key in a1_two_faults_dict.keys(): # one fault on ancilla 2
                final_error = a2_residual_error_dict[j]
                if final_error.sum() > 4:
                    return False
                    print("3 faults on A1, 1 fault on A2, final error weight", final_error.sum())
            if key in a2_two_faults_dict.keys(): # one fault on ancilla 1
                final_error = a1_residual_error_dict[i]
                if final_error.sum() > 4:
                    return False
                    print("3 faults on A2, 1 fault on A1, final error weight", final_error.sum())
      
    return True

# |0> FT preparation
flip_type = 0 # 0 for X-type, 1 for Z-type
PB = [(2,6),(5,1),(6,0),(0,5),(4,2),(0,3),(1,4)] 
PC = [(3,1),(0,2),(2,6),(6,4),(5,0),(6,5),(3,6)] # lack 3
PD = [(5,3),(6,1),(1,2),(2,5),(4,0),(3,4),(4,5)] # lack 6
list_prod = lambda A : reduce(operator.matmul, [Eij(a[0],a[1]) for a in A], np.eye(n, dtype=int)) % 2

for _ in range(10):
    for a in list(itertools.permutations([0,1,2,3,3,6,6])):
        PA = [(ce(a[0]),a[0]),(ce(a[1]),a[1]),(ce(a[2]),a[2]),(ce(a[3]),a[3]),(ce(a[4]),a[4]),(ce(a[5]),a[5]),(ce(a[6]),a[6])]
        # print("testing", PA)
        pass_test = test_faults(list_prod(PA) @ list_prod(PB[::-1]) % 2, flip_type)
        if not pass_test: continue
        print("pass AB", PA, flush=True)
        pass_test = test_faults(list_prod(PA) @ list_prod(PC[::-1]) % 2, flip_type)
        if not pass_test: continue
        print("pass AC", PA, flush=True)
        pass_test = test_faults(list_prod(PA) @ list_prod(PD[::-1]) % 2, flip_type)
        if not pass_test: continue
        print("pass ALL", PA, flush=True)