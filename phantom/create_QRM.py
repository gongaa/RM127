import numpy as np
from src.codes_q import css_code, kernel , inverse
from src.utils import format_np_array_compact 
from src.trans_Z_rot import *

m = 3
N = 2 ** m
l = 1 # choose a number between [1,m-1]
# phantom QRM parameter [[2^m, m-l+1, 2^{m-l} / 2^l]]

bin_wt = lambda i: bin(i)[2:].count('1')
bit_rev = lambda t: int(bin(t)[2:].rjust(m, '0')[::-1], 2)

int2bin = lambda i: [int(c) for c in bin(i)[2:].rjust(m, '0')]
bin2int = lambda l: int(''.join(map(str, l)), 2)

puncture = True
if puncture: l = 1
F = np.array([[1,0],[1,1]])
E = F
for i in range(m-1):
    E = np.kron(E, F)

# X stabilizer generators: all degree < l monomials
X_stab_row_indices = [N-1-i for i in range(N) if bin_wt(i) < l]
# extra_X_stab_indices = ["110001", "101001", "100101", "011001", "010101", "001101", "111000", "110100", "101100", "011100"] # another option
# extra_X_stab_indices = ["110010", "101010", "100110", "011010", "010110", "001110", "111000", "110100", "101100", "011100"]
# extra_X_row_indices = [N-1-bin2int(idx) for idx in extra_X_stab_indices]
# X_stab_row_indices += extra_X_row_indices
# print("extra X stab indices", extra_X_row_indices)
Hx = E[X_stab_row_indices]
# X logical generators: x_1...x_{l-1}x_l, ... , x_1...x_{l-1}x_{m}
X_logical_indices = []
for i in range(l-1, m):
    idx = list("1"*(l-1)+"0"*(m-l+1))
    idx[i] = "1"
    X_logical_indices.append("".join(idx)[::-1])
print("X logical indices", X_logical_indices)
X_logical_row_indices = [N-1-bin2int(idx) for idx in X_logical_indices]
Lx = E[X_logical_row_indices]
# Z logical generators:
Lz = E.T[X_logical_row_indices]


# row index from bottom to top: 000...0 to 111...1

if puncture:
    Lx = Lx[:,:-1]
    Hx = Hx[:,:-1]

k = Lx.shape[0]
def Eij(i,j):
    A = np.eye(k, dtype=int)
    A[i,j] = 1
    return A

# for the [[16,6,4]] code, set m = 4
# X_stab_row_indices = [N-1-i for i in range(N) if bin_wt(i) < 2]
# Hx = E[X_stab_row_indices]
# X_logical_row_indices = [N-1-i for i in range(N) if bin_wt(i) == 2]
# Lx = E[X_logical_row_indices]

# for the [[15,7,3]] code,
# E = E[:,:-1]
# X_stab_row_indices = [N-1-i for i in range(N) if bin_wt(i) == 1]
# Hx = E[X_stab_row_indices]
# X_logical_row_indices = [N-1-i for i in range(N) if bin_wt(i) == r or bin_wt(i)==0]
# Lx = E[X_logical_row_indices]

print("Lx")
print(format_np_array_compact(Lx))
print("Lz")
print(format_np_array_compact(Lz))
print("Hx")
print(format_np_array_compact(Hx))

mat = np.vstack((Lx, Hx))
Hz = kernel(mat)[0]
code = css_code(Hx, Hz)
Lz = code.lz
Hz = code.hz
print(f"Lx shape: {Lx.shape}, Hx shape: {Hx.shape}, Lz shape: {Lz.shape}, Hz shape: {Hz.shape}")
print(transversal_Z_rotation_all(Lx, Hx, 2))
print(transversal_Z_rotation_all(Lx, Hx, 3))
print(transversal_Z_rotation_all(Lz, Hz, 2))
print(transversal_Z_rotation_all(Lz, Hz, 3))

######### concat with [[2,1,dx=1,dz=2]] phase-flip repetition code ################
concat_Hx = []
N = Lx.shape[1]
for row in Hx:
    temp_row = []
    for i in range(N):
        temp_row += [row[i], 0]
    concat_Hx.append(temp_row)
for row in np.eye(N, dtype=int):
    temp_row = []
    for i in range(N):
        temp_row += [row[i], row[i]]
    concat_Hx.append(temp_row)
concat_Lx = []
for row in Lx:
    temp_row = []
    for i in range(N):
        temp_row += [row[i], 0]
    concat_Lx.append(temp_row)
Hx = np.array(concat_Hx)
Lx = np.array(concat_Lx)
print("X logical shape", Lx.shape)
###################################################################################

mat = np.vstack((Lx, Hx))
Hz = kernel(mat)[0]
print("X stabilizer shape", Hx.shape)
print("Z stabilizer shape", Hz.shape)
code = css_code(Hx, Hz)
print(f"[[{code.N},{code.K}]]")

print("Lx")
print(code.lx)
print("Hx")
print(code.hx)

print(transversal_Z_rotation_all(code.lx, code.hx, 2))
print(transversal_Z_rotation_all(code.lz, code.hz, 2))
print(transversal_Z_rotation_all(code.lx, code.hx, 3))
print(transversal_Z_rotation_all(code.lz, code.hz, 3))
fold_diagonal_gate(code.lx, code.hx, cList=[(0,1),(2,3),(4,5),(6,7),(8,9),(10,11),(12,13)])

################## comment out this block if puncture=True or concat with phase-flip repetition code ##############
if puncture != True:
    code.lx = Lx
    code.lx_hx = np.vstack((Lx, code.hx))
    code.lz = Lz
    code.lz_hz = np.vstack((Lz, code.hz))

################# for [[64,4,8]] #######################
if m != 6: exit()
permutation = [i for i in range(code.N)]
for i in range(code.N):
    i_bin = int2bin(i)
    # i_bin = [1-i_bin[5], 1-i_bin[4], 1-i_bin[3], 1-i_bin[2], 1-i_bin[1], 1-i_bin[0]]    # SS
    i_bin = [1-i_bin[5], 1-i_bin[4], 1-i_bin[2], 1-i_bin[3], 1-i_bin[1], 1-i_bin[0]]    # CZ
    permutation[i] = bin2int(i_bin)
fixed_point = [i for (idx, i) in enumerate(permutation) if i==idx]
print("fixed points", fixed_point)
# need to check INVOLUTION !
for i in range(code.N):
    j = permutation[i]
    if i != j:
        assert permutation[j] == i, "Not an involution!"

# implement mapping by a fold-S gate
# want to see if a mapped result is a stabilizer, or is a logical
# if it is a logical, what are the contributions from the basis
for X_stab in code.hx:
    op = X_stab[permutation]
    assert code.is_Z_stabilizer(op)

for X_log in code.lx:
    op = X_log[permutation]
    print("X logical mapped to Z logical?", code.is_Z_logical(op))
