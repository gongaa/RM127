cimport cython
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "Decoder/Decoder_polar.hpp":
    cdef cppclass Decoder_polar_SCL:
        Decoder_polar_SCL(const int& K, const int& N, const int& L, const vector[bool]& frozen_bits)
        int decode(const double *Y_N, int *V_K)
        int get_last_info_bit()



cdef class PyDecoder_polar_SCL:
    cdef Decoder_polar_SCL* SCL_decoder
    cdef int m, r, N, K, list_size
    cdef int num_flip
    cdef int last_info_bit
    cdef int MEM_ALLOCATED
    cdef vector[bool] frozen_bits
    cdef double* llr_noisy_codeword
    cdef int* noisy_codeword
    cdef int* SCL_denoised_codeword
    cpdef int decode(self, list nnz) 
    cdef int count_weight(self, int i)
    cdef void count_flip(self)