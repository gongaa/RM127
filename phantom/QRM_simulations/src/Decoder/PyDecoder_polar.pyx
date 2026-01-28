from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdlib cimport malloc, free
from PyDecoder_polar cimport Decoder_polar_SCL

cdef class PyDecoder_polar_SCL:

	def __cinit__(self, int r, int m, int state=-1):
		self.MEM_ALLOCATED = False
		self.m = m
		print("decoder m =", m, ", r =", r)
		# phantom QRM paramenter [[2^m, m-r+1, 2^{m-r} / 2^r]]
		self.N = int(2 ** m)
		self.list_size = 16 
		self.r = r
		self.K = m-r+1 # depends on order r
		self.X_frozen_bits.resize(self.N, 0)
		self.Z_frozen_bits.resize(self.N, 0)
		self.X_info_indices.resize(self.K, 0) # which rows to read
		self.Z_info_indices.resize(self.K, 0)
		self.X_info_bits.resize(self.K, 0) # what are the values in these rows
		self.Z_info_bits.resize(self.K, 0)
		self.llr_noisy_codeword_X = <double*>malloc(self.N * sizeof(double))
		self.noisy_codeword_X = <int*>malloc(self.N * sizeof(int))
		self.SCL_denoised_codeword_X = <int*>malloc(self.N * sizeof(int))
		self.llr_noisy_codeword_Z = <double*>malloc(self.N * sizeof(double))
		self.noisy_codeword_Z = <int*>malloc(self.N * sizeof(int))
		self.SCL_denoised_codeword_Z = <int*>malloc(self.N * sizeof(int))

		cdef int bias = 0
		cdef int row = 0
		for i in range(r-1):
			bias += 1<<i
		for i in range(r-1, m):
			row = bias + (1<<i)
			self.Z_info_indices[i-r+1] = row
			self.X_info_indices[i-r+1] = self.N-1-row
		self.X_info_indices = sorted(self.X_info_indices)
		self.Z_info_indices = sorted(self.Z_info_indices)

		cdef int weight
		cdef bool found = False
		for i in range(self.N):
			weight = self.count_weight(self.N-1-i)
			if weight < r: # is an X stabilizer, hence frozen for Z
				self.Z_frozen_bits[self.N-1-i] = 1
			elif weight > r: # is an Z stabilizer, hence frozen for X
				self.X_frozen_bits[i] = 1
			# elif not (i in self.X_info_indices): # Z stabilizer, hence frozen for X
			else:
				found = False
				for j in range(self.K):
					if self.X_info_indices[j] == i:
						found = True
						break
				if not found: # Z stabilizer, hence frozen for X
					self.X_frozen_bits[i] = 1
				elif state == 1: # all-plus state; set logical to X stabilizer, i.e., frozen for Z
					self.Z_frozen_bits[self.N-1-i] = 1
				elif state == 0: # all-zero state; set logical to Z stabilizer, i.e., frozen for X
					self.X_frozen_bits[i] = 1

		self.SCL_decoder_X = new Decoder_polar_SCL(self.K, self.N, self.list_size, self.X_frozen_bits, self.X_info_indices)
		self.SCL_decoder_Z = new Decoder_polar_SCL(self.K, self.N, self.list_size, self.Z_frozen_bits, self.Z_info_indices)
		self.MEM_ALLOCATED = True
		self.num_X_flip = 0
		self.num_Z_flip = 0
		# print("Z frozen pattern", self.Z_frozen_bits)
		# print("X frozen pattern", self.X_frozen_bits)
		cdef int num_Z_frozen = 0
		for i in range(self.N):
			if self.Z_frozen_bits[i]: 
				num_Z_frozen += 1
		# print("number of Z frozen bits", num_Z_frozen)
		cdef int num_X_frozen = 0
		for i in range(self.N): 
			if self.X_frozen_bits[i]: 
				num_X_frozen += 1
		# print("number of X frozen bits", num_X_frozen)
		if state == -1:
			assert (num_X_frozen + num_Z_frozen + self.K) == self.N, "arbitary state: number of frozen bits do not match"
		else:
			assert (num_X_frozen + num_Z_frozen) == self.N, "all-plus or all-zero: number of frozen bits do not match"
		# print("X info indices", self.X_info_indices)
		# print("Z info indices", self.Z_info_indices)
		

	def __dealloc__(self):
		if self.MEM_ALLOCATED:
			del self.SCL_decoder_X
			del self.SCL_decoder_Z
			free(self.llr_noisy_codeword_X)
			free(self.noisy_codeword_X)
			free(self.SCL_denoised_codeword_X)
			free(self.llr_noisy_codeword_Z)
			free(self.noisy_codeword_Z)
			free(self.SCL_denoised_codeword_Z)

	@property
	def num_X_flip(self):
		return self.num_X_flip

	@property
	def num_Z_flip(self):
		return self.num_Z_flip
	
	@property
	def X_info_bits(self): # indicating which X logical class it belongs
		return self.X_info_bits

	@property
	def Z_info_bits(self): # indicating which Z logical class it belongs
		return self.Z_info_bits

	@property
	def X_correction(self):
		self.X_correction = []
		for i in range(self.N):
			if self.noisy_codeword_X[i] != self.SCL_denoised_codeword_X[i]:
				self.X_correction.append(i)
		return self.X_correction
	
	@property
	def Z_correction(self):
		self.Z_correction = []
		for i in range(self.N):
			if self.noisy_codeword_Z[i] != self.SCL_denoised_codeword_Z[i]:
				self.Z_correction.append(self.N-1-i)
		return self.Z_correction[::-1]

	cdef int count_weight(self, int i):
		# Count the number of ones in the binary representation
		cdef int cnt = 0
		while i:
			cnt += i & 1
			i >>= 1
		return cnt

	cdef void count_X_flip(self):
		self.num_X_flip = 0
		for i in range(self.N):
			if self.noisy_codeword_X[i] != self.SCL_denoised_codeword_X[i]:
				self.num_X_flip += 1

	cdef void count_Z_flip(self):
		self.num_Z_flip = 0
		for i in range(self.N):
			if self.noisy_codeword_Z[i] != self.SCL_denoised_codeword_Z[i]:
				self.num_Z_flip += 1

	cpdef int decode_X_flip(self, list nnz):
		for i in range(self.N):
			self.noisy_codeword_X[i] = 0
		for i in range(self.N):
			self.llr_noisy_codeword_X[i] = 1.0 # or use soft information here
		for i in nnz:
			self.noisy_codeword_X[i] = 1
			self.llr_noisy_codeword_X[i] = -1.0 # or use soft information here

		self.SCL_decoder_X.decode(self.llr_noisy_codeword_X, self.SCL_denoised_codeword_X)

		self.count_X_flip()

		return self.num_X_flip

	cpdef int decode_Z_flip(self, list nnz):
		for i in range(self.N):
			self.noisy_codeword_Z[i] = 0
		for i in range(self.N):
			self.llr_noisy_codeword_Z[i] = 1.0 # or use soft information here
		for i in nnz:
			self.noisy_codeword_Z[self.N-1-i] = 1
			self.llr_noisy_codeword_Z[self.N-1-i] = -1.0 # or use soft information here

		self.SCL_decoder_Z.decode(self.llr_noisy_codeword_Z, self.SCL_denoised_codeword_Z)

		self.count_Z_flip()

		return self.num_Z_flip