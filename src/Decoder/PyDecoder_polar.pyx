from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from PyDecoder_polar cimport Decoder_polar_SCL

cdef class PyDecoder_polar_SCL:

	def __cinit__(self, int r, int m=7):
		self.MEM_ALLOCATED = False
		self.m = m
		self.N = int(2 ** m)
		self.list_size = 8
		self.K = 0 # depends on order r
		self.r = r

		self.frozen_bits.resize(self.N, 1)
		self.llr_noisy_codeword = <double*>malloc(self.N * sizeof(double))
		self.noisy_codeword = <int*>malloc(self.N * sizeof(int))
		self.SCL_denoised_codeword = <int*>malloc(self.N * sizeof(int))

		cdef int weight
		for i in range(self.N):
			weight = self.count_weight(i)
			if weight >= (self.m - self.r):
				self.frozen_bits[i] = 0 # not frozen
				self.K += 1

		self.SCL_decoder = new Decoder_polar_SCL(self.K, self.N, self.list_size, self.frozen_bits)
		self.MEM_ALLOCATED = True
		self.num_flip = 0
		self.last_info_bit = 0
		

	def __dealloc__(self):
		if self.MEM_ALLOCATED:
			del self.SCL_decoder
			free(self.llr_noisy_codeword)
			free(self.noisy_codeword)
			free(self.SCL_denoised_codeword)

	@property
	def num_flip(self):
		return self.num_flip

	@property
	def last_info_bit(self): # indicating which class it belongs, C or 1+C, C is the weight even subcode
		return self.last_info_bit

	cdef int count_weight(self, int i):
		# Count the number of ones in the binary representation
		cdef int cnt = 0
		while i:
			cnt += i & 1
			i >>= 1
		return cnt

	cdef void count_flip(self):
		self.num_flip = 0
		for i in range(self.N-1): # do not count the punctured bit
			if self.noisy_codeword[i] != self.SCL_denoised_codeword[i]:
				self.num_flip += 1

	cpdef int decode(self, list nnz):
		for i in range(self.N):
			self.noisy_codeword[i] = 0
		for i in range(self.N):
			self.llr_noisy_codeword[i] = 1.0 # or use soft information here
		for i in nnz:
			self.noisy_codeword[i] = 1
			self.llr_noisy_codeword[i] = -1.0 # or use soft information here

		self.llr_noisy_codeword[self.N-1] = 0 # last bit is punctured (erased)

		self.SCL_decoder.decode(self.llr_noisy_codeword, self.SCL_denoised_codeword)

		self.last_info_bit = self.SCL_decoder.get_last_info_bit()

		self.count_flip()

		return self.num_flip

