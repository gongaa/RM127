# Fault tolerant preparation of Quantum Reed Muller codes of blocklength 127 

This repo contains the source code of the paper "Computation with Quantum Reed-Muller codes and their mapping to 2D atom array". 



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
`structured_test.py` contains global MITM testing/counting across four patches up to order six. `strict_FT/` stores all the logs. The following runtime estimate are based on my Linux workstation (Intel i9-13900K, 64GB memory). Order up to four takes less than half a minute.

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


