NVCC := /usr/local/cuda/bin/nvcc
NVCC_OPTS := -arch=sm_30
sudokusolver: sudokusolver.cu
	$(NVCC) $(NVCC_OPTS) sudokusolver.cu -o sudokusolver
