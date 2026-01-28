cimport cython
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "Decoder/Decoder_polar.hpp":
    cdef cppclass Decoder_polar_SCL:
        Decoder_polar_SCL(const int& K, const int& N, const int& L, const vector[bool]& frozen_bits, const vector[int]& info_indices)
        int decode(const double *Y_N, int *V_K)
        void get_info_bits(vector[bool]& Y_K)



cdef class PyDecoder_polar_SCL:
    cdef Decoder_polar_SCL* SCL_decoder_X
    cdef Decoder_polar_SCL* SCL_decoder_Z
    cdef int m, r, N, K, list_size
    cdef int num_X_flip, num_Z_flip
    cdef list X_correction
    cdef list Z_correction

    cdef int MEM_ALLOCATED
    cdef vector[bool] X_frozen_bits
    cdef vector[bool] Z_frozen_bits
    cdef vector[int] X_info_indices
    cdef vector[int] Z_info_indices
    cdef vector[bool] X_info_bits
    cdef vector[bool] Z_info_bits
    cdef double* llr_noisy_codeword_X
    cdef int* noisy_codeword_X
    cdef int* SCL_denoised_codeword_X
    cdef double* llr_noisy_codeword_Z
    cdef int* noisy_codeword_Z
    cdef int* SCL_denoised_codeword_Z
    cpdef int decode_X_flip(self, list nnz) 
    cpdef int decode_Z_flip(self, list nnz) 
    cdef void count_X_flip(self)
    cdef void count_Z_flip(self)
    cdef int count_weight(self, int i)