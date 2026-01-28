#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions and layers for the FEC package."""

import numpy as np
import shutil, subprocess

def bin2int(arr):
    """Convert binary array to integer.

    For example ``arr`` = `[1, 0, 1]` is converted to `5`.

    Input
    -----
        arr: int or float
            An iterable that yields 0's and 1's.

    Output
    -----
        : int
            Integer representation of ``arr``.

    """
    if len(arr) == 0: return None
    return int(''.join([str(x) for x in arr]), 2)

def int2bin(num, len_):
    """
    Convert ``num`` of int type to list of length ``len_`` with 0's and 1's.
    ``num`` and ``len_`` have to non-negative.

    For e.g., ``num`` = `5`; `int2bin(num`, ``len_`` =4) = `[0, 1, 0, 1]`.

    For e.g., ``num`` = `12`; `int2bin(num`, ``len_`` =3) = `[1, 0, 0]`.

    Input
    -----
        num: int
            An integer to be converted into binary representation.

        len_: int
            An integer defining the length of the desired output.

    Output
    -----
        : list of int
            Binary representation of ``num`` of length ``len_``.
    """
    assert num >= 0,  "Input integer should be non-negative"
    assert len_ >= 0,  "width should be non-negative"

    bin_ = format(num, f'0{len_}b')
    binary_vals = [int(x) for x in bin_[-len_:]] if len_ else []
    return binary_vals

def alist2mat(alist, verbose=True):
    # pylint: disable=line-too-long
    r"""Convert `alist` [MacKay]_ code definition to `full` parity-check matrix.

    Many code examples can be found in [UniKL]_.

    About `alist` (see [MacKay]_ for details):

        - `1.` Row defines parity-check matrix dimension `m x n`
        - `2.` Row defines int with `max_CN_degree`, `max_VN_degree`
        - `3.` Row defines VN degree of all `n` column
        - `4.` Row defines CN degree of all `m` rows
        - Next `n` rows contain non-zero entries of each column (can be zero padded at the end)
        - Next `m` rows contain non-zero entries of each row.

    Input
    -----
    alist: list
        Nested list in `alist`-format [MacKay]_.

    verbose: bool
        Defaults to True. If True, the code parameters are printed.

    Output
    ------
    (pcm, k, n, coderate):
        Tuple:

    pcm: ndarray
        NumPy array of shape `[n-k, n]` containing the parity-check matrix.

    k: int
        Number of information bits.

    n: int
        Number of codewords bits.

    coderate: float
        Coderate of the code.

    Note
    ----
        Use :class:`~sionna.fec.utils.load_alist` to import alist from a
        textfile.

        For example, the following code snippet will import an alist from a file called ``filename``:

        .. code-block:: python

            al = load_alist(path = filename)
            pcm, k, n, coderate = alist2mat(al)
    """

    assert len(alist)>4, "Invalid alist format."

    n = alist[0][0]
    m = alist[0][1]
    v_max = alist[1][0]
    c_max = alist[1][1]
    k = n - m
    coderate = k / n

    vn_profile = alist[2]
    cn_profile = alist[3]

    # plausibility checks
    assert np.sum(vn_profile)==np.sum(cn_profile), "Invalid alist format."
    assert np.max(vn_profile)==v_max, "Invalid alist format."
    assert np.max(cn_profile)==c_max, "Invalid alist format."

    if len(alist)==len(vn_profile)+4:
        print("Note: .alist does not contain (redundant) CN perspective.")
        print("Recovering parity-check matrix from VN only.")
        print("Please verify the correctness of the results manually.")
        vn_only = True
    else:
        assert len(alist)==len(vn_profile) + len(cn_profile) + 4, \
                                                "Invalid alist format."
        vn_only = False

    pcm = np.zeros((m,n))
    num_edges = 0 # count number of edges

    for idx_v in range(n):
        for idx_i in range(vn_profile[idx_v]):
            # first 4 rows of alist contain meta information
            idx_c = alist[4+idx_v][idx_i]-1 # "-1" as this is python
            pcm[idx_c, idx_v] = 1
            num_edges += 1 # count number of edges (=each non-zero entry)

    # validate results from CN perspective
    if not vn_only:
        for idx_c in range(m):
            for idx_i in range(cn_profile[idx_c]):
                # first 4 rows of alist contain meta information
                # follwing n rows contained VN perspective
                idx_v = alist[4+n+idx_c][idx_i]-1 # "-1" as this is python
                assert pcm[idx_c, idx_v]==1 # entry must already exist

    if verbose:
        print("Number of variable nodes (columns): ", n)
        print("Number of check nodes (rows): ", m)
        print("Number of information bits per cw: ", k)
        print("Number edges: ", num_edges)
        print("Max. VN degree: ", v_max)
        print("Max. CN degree: ", c_max)
        print("VN degree: ", vn_profile)
        print("CN degree: ", cn_profile)

    return pcm, k, n, coderate

def load_alist(path):
    """Read `alist`-file [MacKay]_ and return nested list describing the
    parity-check matrix of a code.

    Many code examples can be found in [UniKL]_.

    Input
    -----
    path:str
        Path to file to be loaded.

    Output
    ------
    alist: list
        A nested list containing the imported alist data.
    """

    alist = []
    with open(path, "r") as reader: # pylint: disable=unspecified-encoding
        # read list line by line (different length)
        for line in reader:
            l = []
            # append all entries
            for word in line.split():
                l.append(int(word))
            if l: # ignore empty lines
                alist.append(l)

    return alist

def make_systematic(mat, is_pcm=False):
    r"""Bring binary matrix in its systematic form.

    Input
    -----
    mat : ndarray
        Binary matrix to be transformed to systematic form of shape `[k, n]`.

    is_pcm: bool
        Defaults to False. If true, ``mat`` is interpreted as parity-check
        matrix and, thus, the last k columns will be the identity part.

    Output
    ------
    mat_sys: ndarray
        Binary matrix in systematic form, i.e., the first `k` columns equal the
        identity matrix (or last `k` if ``is_pcm`` is True).

    column_swaps: list of int tuples
        A list of integer tuples that describes the swapped columns (in the
        order of execution).

    Note
    ----
    This algorithm (potentially) swaps columns of the input matrix. Thus, the
    resulting systematic matrix (potentially) relates to a permuted version of
    the code, this is defined by the returned list ``column_swap``.
    Note that, the inverse permutation must be applied in the inverse list
    order (in case specific columns are swapped multiple times).

    If a parity-check matrix is passed as input (i.e., ``is_pcm`` is True), the
    identity part will be re-arranged to the last columns."""

    m = mat.shape[0]
    n = mat.shape[1]

    assert m<=n, "Invalid matrix dimensions."

    # check for all-zero columns (=unchecked nodes)
    if is_pcm:
        c_node_deg = np.sum(mat, axis=0)
        if np.any(c_node_deg==0):
            warnings.warn("All-zero column in parity-check matrix detected. " \
                "It seems as if the code contains unprotected nodes.")

    mat = np.copy(mat)
    column_swaps = [] # store all column swaps

    # convert to bool for faster arithmetics
    mat = mat.astype(bool)

    # bring in upper triangular form
    for idx_c in range(m):
        success = False
        # step 1: find next leading "1"
        for idx_r in range(idx_c,m):
            # skip if entry is "0"
            if mat[idx_r, idx_c]:
                mat[[idx_c, idx_r]] = mat[[idx_r, idx_c]] # swap rows
                success = True
                break

        # Could not find "1"-entry for column idx_c
        # => swap with columns from non-sys part
        # The task is to find a column with index idx_cc that has a "1" at
        # row idx_c
        if not success:
            for idx_cc in range(m, n):
                if mat[idx_c, idx_cc]:
                    # swap columns
                    mat[:,[idx_c, idx_cc]] = mat[:,[idx_cc, idx_c]]
                    column_swaps.append([idx_c, idx_cc])
                    success=True
                    break

        if not success:
            raise ValueError("Could not succeed; mat is not full rank?")

        # we can now assume a leading "1" at row idx_c
        for idx_r in range(idx_c+1, m):
            if mat[idx_r, idx_c]:
                mat[idx_r,:] ^= mat[idx_c,:] # bin. add of row idx_c to idx_r

    # remove upper triangle part in inverse order
    for idx_c in range(m-1, -1, -1):
        for idx_r in range(idx_c-1, -1, -1):
            if mat[idx_r, idx_c]:
                mat[idx_r,:] ^= mat[idx_c,:] # bin. add of row idx_c to idx_r

    # verify results
    assert np.array_equal(mat[:,:m], np.eye(m)), \
                            "Internal error, could not find systematic matrix."

    # bring identity part to end of matrix if parity-check matrix is provided
    if is_pcm:
        im = np.copy(mat[:,:m])
        mat[:,:m] = mat[:,-m:]
        mat[:,-m:] = im
        # and track column swaps
        for idx in range(m):
            column_swaps.append([idx, n-m+idx])

    # return integer array
    mat = mat.astype(int)
    return mat, column_swaps

###########################################################
# Functions adapted from the ldpc package
###########################################################

def row_echelon(mat, reduced=False):
    r"""Converts a binary matrix to (reduced) row echelon form via Gaussian Elimination, 
    also works for rank-deficient matrix. Unlike the make_systematic method,
    no column swaps will be performed.

    Input 
    ----------
    mat : ndarry
        A binary matrix in numpy.ndarray format.
    reduced: bool
        Defaults to False. If true, the reduced row echelon form is returned. 
    
    Output
    -------
    row_ech_form: ndarray
        The row echelon form of input matrix.
    rank: int
        The rank of the matrix.
    transform: ndarray
        The transformation matrix such that (transform_matrix@matrix)=row_ech_form
    pivot_cols: list
        List of the indices of pivot num_cols found during Gaussian elimination
    """

    m, n = np.shape(mat)
    # Don't do "m<=n" check, allow over-complete matrices
    mat = np.copy(mat)
    # Convert to bool for faster arithmetics
    mat = mat.astype(bool)
    transform = np.identity(m).astype(bool)
    pivot_row = 0
    pivot_cols = []

    # Allow all-zero column. Row operations won't induce all-zero columns, if they are not present originally.
    # The make_systematic method will swap all-zero columns with later non-all-zero columns.
    # Iterate over cols, for each col find a pivot (if it exists)
    for col in range(n):
        # Select the pivot - if not in this row, swap rows to bring a 1 to this row, if possible
        if not mat[pivot_row, col]:
            # Find a row with a 1 in this column
            swap_row_index = pivot_row + np.argmax(mat[pivot_row:m, col])
            # If an appropriate row is found, swap it with the pivot. Otherwise, all zeroes - will loop to next col
            if mat[swap_row_index, col]:
                # Swap rows
                mat[[swap_row_index, pivot_row]] = mat[[pivot_row, swap_row_index]]
                # Transformation matrix update to reflect this row swap
                transform[[swap_row_index, pivot_row]] = transform[[pivot_row, swap_row_index]]

        if mat[pivot_row, col]: # will evaluate to True if this column is not all-zero
            if not reduced: # clean entries below the pivot 
                elimination_range = [k for k in range(pivot_row + 1, m)]
            else:           # clean entries above and below the pivot
                elimination_range = [k for k in range(m) if k != pivot_row]
            for idx_r in elimination_range:
                if mat[idx_r, col]:    
                    mat[idx_r] ^= mat[pivot_row]
                    transform[idx_r] ^= transform[pivot_row]
            pivot_row += 1
            pivot_cols.append(col)

        if pivot_row >= m: # no more rows to search
            break

    rank = pivot_row
    row_ech_form = mat.astype(int)

    return [row_ech_form, rank, transform.astype(int), pivot_cols]

def rank(mat):
    r"""Returns the rank of a binary matrix

    Input 
    ----------
    mat: ndarray
        A binary matrix in numpy.ndarray format

    Output
    -------
    int
        The rank of the matrix"""
    return row_echelon(mat)[1]

def kernel(mat):
    r"""Computes the kernel of the matrix M.
    All vectors x in the kernel of M satisfy the following condition::

        Mx=0 \forall x \in ker(M)

    Input 
    ----------
    mat: ndarray
        A binary matrix in numpy.ndarray format.
    
    Output
    -------
    ker: ndarray
        A binary matrix which is the kernel of the input binary matrix.

    rank: int
        Rank of transposed mat, which is the same as the rank of mat.

    pivot_cols: list
        List of the indices of pivot of the transposed mat. Can be used in row_basis.
    
    Note
    -----
    Why does this work?

    The transformation matrix, P, transforms the matrix M into row echelon form, ReM::

        P@M=ReM=[A,0]^T,
    
    where the width of A is equal to the rank. This means the bottom n-k rows of P
    must produce a zero vector when applied to M. For a more formal definition see
    the Rank-Nullity theorem.
    """

    transpose = mat.T
    m, _ = transpose.shape
    _, rank, transform, pivot_cols = row_echelon(transpose)
    ker = transform[rank:m]
    return ker, rank, pivot_cols

def row_basis(mat):
    r"""Outputs a basis for the rows of the matrix.

    Input
    ----------
    mat: ndarray
        The input matrix.

    Output
    -------
    basis: ndarray
        A numpy.ndarray matrix where each row is a basis element."""
    return mat[row_echelon(mat.T)[3]]

def compute_code_distance(mat, is_pcm=True, is_basis=False):
    r'''Computes the distance of the linear code given by the input parity check / generator matrix. 
    The code distance is given by the minimum weight of a nonzero codeword.

    Note
    ----
    The runtime of this function scales exponentially with the block size. In practice, computing the code distance of codes with block lengths greater than ~10 will be very slow.

    Parameters
    ----------
    mat: ndarray
        The parity check matrix
    
    is_pcm: bool
        Defaults to True. If false, mat is interpreted as a generator matrix.
    
    Returns
    -------
    int
        The code distance
    '''
    gen = mat
    if is_pcm:
        gen = kernel(mat)
    if len(gen)==0: return np.inf # infinite code distance
    cw = gen
    if not is_basis:
        cw = row_basis(gen) # nonzero codewords
    return np.min(np.sum(cw, axis=1))

def inverse(mat):
    r"""Computes the left inverse of a full-rank matrix.

    Input
    ----------
    matrix: ndarray
        The binary matrix to be inverted in numpy.ndarray format. This matrix must either be
        square full-rank or rectangular with full-column rank.

    Output
    -------
    inverse: ndarray
        The inverted binary matrix
    
    Note
    -----
    The `left inverse' is computed when the number of rows in the matrix
    exceeds the matrix rank. The left inverse is defined as follows::

        Inverse(M.T@M)@M.T

    We can make a further simplification by noting that the row echelon form matrix
    with full column rank has the form::

        row_echelon_form=P@M=vstack[I,A]

    In this case the left inverse simplifies to::

        Inverse(M^T@P^T@P@M)@M^T@P^T@P=M^T@P^T@P=row_echelon_form.T@P"""

    m, n = mat.shape
    reduced_row_ech, rank, transform, _ = row_echelon(mat, reduced=True)
    if m == n and rank == m:
        return transform
    # compute the left-inverse
    elif m > rank and n == rank:  # left inverse
        return reduced_row_ech.T @ transform % 2
    else:
        raise ValueError("This matrix is not invertible. Please provide either a full-rank square\
        matrix or a rectangular matrix with full column rank.")
        
def Lempel_factorization(A):
    r"""Lempel factorization, given a symmetric binary matrix A, find matrix E such that A=EE^T
        if A has at least a one on the diagonal, E can be a square matrix (same shape as A)
        
    Input
    ------------
    A: ndarray
        The binary symmetric matrix to be factored. Expect A to be full-rank and has 
        at least a one on the diagonal.
        
    Ouput
    ------------
    E: ndarray
        Such that A = EE^T
        
    """
    # Step 1: build elementary factorization
    num_rows, num_cols = A.shape
    assert num_rows == num_cols
    assert not (A - A.T).any()
    N1 = np.where(A.sum(axis=1) % 2)[0] # rows s.t. sum odd
    N2 = [(i,j) for i in range(num_rows) for j in range(num_cols) if i<j and A[i,j]==1]
    E = np.zeros((num_rows, len(N1)+len(N2)), dtype=int)
    # k-columns (1 in row k and zeros elsewhere) according to N1
    for idx, k in enumerate(N1):
        E[k, idx] = 1
    # (i,j)-columns (1 in rows i and j and zeros elsewhere) according to N2
    for idx, (i,j) in enumerate(N2):
        E[i, idx+len(N1)] = 1
        E[j, idx+len(N1)] = 1
        
    while True:
        _, num_cols_E = E.shape
        if num_cols_E == num_rows:
            return E
        # Step 2: find a set of columns whose sum is zero
        kernel_E = kernel(E)[0]
        mask = kernel_E[0] # first row is enough
        Z = E[:,mask.astype(bool)]
        if mask.sum() % 2 == 1: # Z contains odd number of columns
            Z = np.hstack((Z, np.zeros((num_rows, 1), dtype=int))) # stack with all-zero

        F = E[:,(1-mask).astype(bool)]
        x = np.expand_dims(F[:,0] + Z[:,0], 1)
        # Step 3: substitute Z
        Z_hat = (Z + x @ np.ones((1, Z.shape[1]), dtype=int)) % 2
        # Step 4: delete the first column from F and Z_hat respectively
        E = np.hstack((F[:,1:], Z_hat[:,1:]))

def format_np_array_compact(mat):
    s = ""
    for r in mat:
        s += ''.join([str(c) for c in r]) + "\n"
    return s[:-1]

def save_mtx(mat, suffix='X'):
    fh = io.BytesIO()
    sp.io.mmwrite(fh, sparse.csr_matrix(mat))
    with open(f"{mat.shape[1]}_{suffix}.mtx", 'w') as f:
        print(fh.getvalue().decode('utf-8'), file=f, end="")
        
def save_triply_even(SX):
    # from the triply-even X-stabilizers, save both X and Z stabilizers in MTX format
    # the two MTX files will be used in verifying distance with QDistRnd
    n = SX.shape[1]
    print("X stabilizer shape", SX.shape)
    with open(f"{n}.txt", 'w') as f:
        print(format_np_array_compact(SX), file=f, end="")
    save_mtx(SX, suffix='X') # save X stabilizers in MTX format
    mat = np.vstack((np.ones((1,n),dtype=int), SX)) # stack with all-one vector (X-logical) 
    SZ = kernel(mat)[0] # Z stabilizers commute with (X stabilizers + X logical)
    print("Z stabilizer shape", SZ.shape)
    save_mtx(SZ, suffix='Z') # save Z stabilizers in MTX format
    print(f"saved to {n}_X.mtx and {n}_Z.mtx")
    code = css_code(SX, SZ)
    print(f"[[{code.N},{code.K}]]")
    print(format_np_array_compact(code.lz))
    
def save_triortho(mat, start=4):
    #     | G_T |
    # G = | G_0 |
    # G_0 is X-stabilizers (SX), G^{\perp} is Z-stabilizers
    n = mat.shape[1]
    with open(f"{n}.txt", 'w') as f:
        print(format_np_array_compact(mat), file=f, end="")
    # wt_even_rows = np.where(np.sum(mat, axis=1) % 2 == 0)[0]
    # SX = mat[wt_even_rows]
    SX = mat[start:]
    SZ = kernel(mat)[0]
    print("X stabilizer shape", SX.shape)
    print("Z stabilizer shape", SZ.shape)
    save_mtx(SX, suffix='X')
    save_mtx(SZ, suffix='Z')
    print(f"saved to {n}_X.mtx and {n}_Z.mtx")
    code = css_code(SX, SZ)
    print(f"[[{code.N},{code.K}]]")
    # print(format_np_array_compact(code.lz)) # print logical Z operators, useful for distance upper bound
    
def keep_full_rank(mat):
    temp = mat[0]
    current_rank = 1
    for row in mat[1:]:
        if rank(np.vstack((temp, row))) > current_rank:
            current_rank += 1
            temp = np.vstack((temp, row))
    return temp

def test_triply_even(mat, no_print=False):
    num_row, num_col = mat.shape
    num_violated = 0
    occur_dict = {}
    for i in range(num_row): occur_dict[i] = 0
    for i in range(num_row):
        for j in range(i, num_row):
            for k in range(j, num_row):
                wt = np.logical_and(mat[i], np.logical_and(mat[j], mat[k])).astype(int).sum()
                if wt % 2:
                    if not no_print:
                        print(f"row {i} {j} {k} violate triply-even")
                    occur_dict[i] += 1; occur_dict[j] += 1; occur_dict[k] += 1
                    num_violated += 1

    print("number of violations", num_violated)
    if not no_print:
        print("sorted occurrence dict", dict(sorted(occur_dict.items(), key=lambda item: item[1])))

def test_wt_even(mat, print_row=False):
    num_row, num_col = mat.shape
    num_odd = 0
    for i in range(num_row):
        wt = mat[i].sum()
        if wt % 2 and print_row:
            print(f"row {i} weight {wt} is odd")
            print(mat[i])
            num_odd += 1
    return num_odd

def test_doubly_even(mat):
    num_row, num_col = mat.shape
    for i in range(num_row):
        for j in range(i+1, num_row):
            wt = np.logical_and(mat[i], mat[j]).astype(int).sum()
            if wt % 4:
                print(f"row {i} and {j} violate doubly-even, intersection wt {wt} not divisible by 4")
                

def get_distance_qdistrnd(mat):
    ncols = mat.shape[1]
    shutil.copyfile(f'{ncols}_X.mtx', f'/Users/anqigong/Downloads/gap-4.13.1/pkg/qdistrnd/matrices/{ncols}_X.mtx')
    shutil.copyfile(f'{ncols}_Z.mtx', f'/Users/anqigong/Downloads/gap-4.13.1/pkg/qdistrnd/matrices/{ncols}_Z.mtx')

    # Start GAP in quiet mode. Call QDistRnd for distance.
    gap_process = subprocess.Popen(
        ["/Users/anqigong/Downloads/gap-4.13.1/gap", '-b'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    #     bufsize=100000
    )

    # Load the module once
    gap_process.stdin.write('LoadPackage("QDistRnd");;\n')
    gap_process.stdin.flush()
    print("GAP says:", gap_process.stdout.readline())

    gap_process.stdin.write('filedir:=DirectoriesPackageLibrary("QDistRnd","matrices");;\n')
    gap_process.stdin.flush()

    cmd = 'lisX:=ReadMTXE(Filename(filedir, "' + f"{ncols}"+ '_X.mtx"),0);;GX:=lisX[3];;'
    cmd += 'lisZ:=ReadMTXE(Filename(filedir, "' + f"{ncols}"+ '_Z.mtx"),0);;GZ:=lisZ[3];;'
    cmd += 'DistRandCSS(GX,GX,100,1,0:field:=GF(2));\n'
    # print(cmd)
    gap_process.stdin.write(cmd)
    gap_process.stdin.flush()

    output_line = gap_process.stdout.readline()
    print(output_line)