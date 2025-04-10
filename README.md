# Fault tolerant preparation of Quantum Reed Muller codes of blocklength 127 

This repo contains the source code of the paper ["Computation with Quantum Reed-Muller codes and their mapping to 2D atom array"](https://arxiv.org/abs/2410.23263).

[Update Apr-10-2025]
* A [note](https://github.com/gongaa/RM127/tree/main/efficiency.pdf) on logical Clifford Synthesis on $K>1$ codes, after reading ["Computing Efficiently in QLDPC Codes"](https://arxiv.org/abs/2502.07150). I interpret their work as follows: "For any [[N,K]] CSS codes, if one can efficiently implement all the logical Clifford gates on one patch, then up to certain assumptions, logical Clifford gates on more patches can be implemented efficiently as well". This result is regardless of the codes (as long as it is CSS) and the implementation of in-block Clifford gates (automorphism, surgery, homomorphic measurement, distillation+teleportation etc., as long as it is efficient). The proof only requires simple linear algebra.\
However, we should be more careful about how "efficiency" is defined (worst-case Clifford depth $O(k)$). Maybe worst-case Clifford (in between non-Clifford gates) is not usually the case in practically-relevant algorithms (like [factoring](https://github.com/strilanc/falling-with-style))? 
* A note on $K=1$ triply-even codes (all $X$-stabilizers have weight divisible by $8$, thus admitting strongly transversal $T$ gate). In particular, I found two doubly-even self dual codes $[[41,1,9]]$ and $[[65,1,13]]$, when used in the doubling transform, triply even codes of parameters $[[177,1,9]]$, $[[271,1,11]]$, $[[401,1,13]]$, $[[559,1,15]]$ can be constructed. They do not need diagonal corrections (S, CZ gates) after transversal $T$, and their blocklengths are slightly better than those in ["Transversal Clifford and T-gate codes of short length and high distance"](https://arxiv.org/abs/2408.12752).\
The $X$ and $Z$ stabilizers of the doubly/triply-even codes can be found in `triply-even/`. See the [README.md](https://github.com/gongaa/RM127/tree/main/triply-even/README.md) there for more info.\
These codes, though bear some interest in theory, are not practically relevant (at least for now). I worked on them to understand the qubit overhead of punctured QRM codes admitting $T$ (e.g. $[[127,1,7]], [[1024,1,15]]$) compared to them.


## Installation 
Please use Python $\geq$ 3.12 and install Cython.
```
make
python setup.py build_ext --inplace
```
Building C++ program via `make` also works on MAC M2 chip, but I only got Cython binding working on Linux machines.

# Data and log availability
The gathered ancilla and exRec simulation logs are available on [Zenodo](https://zenodo.org/records/14003891), you can download them as follows.
```
wget https://zenodo.org/records/14003891/files/logs_exRec.zip && unzip logs_exRec.zip && rm -rf __MACOSX
wget https://zenodo.org/records/14003891/files/logs_prep_SPAM_half_CNOT.zip && unzip logs_prep_SPAM_half_CNOT.zip
wget https://zenodo.org/records/14003891/files/logs_prep_SPAM_equal_CNOT.zip && unzip logs_prep_SPAM_equal_CNOT.zip
rm *.zip
```
These ancilla took supercomputer Euler twenty days to simulate, might be useful if one wants to run magic state distillation in the future. Please refer to `utils.py` and `full_Steane.py` for loading these ancilla. The propagation dictionaries are already contained there, but you can generate new ones following the comments in `full_prep_sim_*.py`.

`logs_exRec/` contains the exRec simulation results, it also contains some useful data gathering scripts.
## Run scripts
Please do `mv run_scipts/* .` before using the scripts to submit jobs to Slurm servers.
* `run_prep.py` - for `full_prep_sim_*.py` to simulate ancilla preparation and store residual errors
* `run_exRec.py` - for `full_Steane.py` to simulate exRec
* `run_SCL.py` - for simulating data qubit noise decoding
* `run_test_pairs.py` - for doing heuristic search of permutations

## Meet-in-the-middle FT testing and malignant set counting
`structured_test.py` contains global MITM testing/counting across four patches up to order six. `strict_FT/` stores all the logs. The following runtime estimates are based on my Linux workstation (Intel i9-13900K, 64GB memory). Order up to four takes less than half a minute.

Order five and six faults will store many order-three fault dictionaries on disk. When loading them back to do MITM, peak memory usage exceeds 50GB.
* state0_Z and state+_X store 68G on disk, runtime ~6h
* state0_X and state+_Z store 29G on disk, runtime ~1.5h
## Other relevant files 
    .
    ├── layout.py                              # for drawing things related to 2D hypercube layout
    ├── test_degeneracy.py                     # test if strict FT is violated in preparation simulations
    ├── utils.py                               # ancilla residual error loader
    └── src                   
        ├── Decoder
        │   ├── Decoder_polar.cpp              # C++ SCL decoder
        │   └── PyDecoder_polar.pyx            # Cython binding for decoder
        └── Simulation         
            └── Simulation_RM127.cpp           # data qubit noise decoding


