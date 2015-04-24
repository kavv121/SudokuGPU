NVCC := /usr/local/cuda/bin/nvcc
NVCC_OPTS := -arch=sm_35 -rdc=true -O2
sudokusolver: sudokusolver.cu
	$(NVCC) $(NVCC_OPTS) sudokusolver.cu -lcudadevrt -o sudokusolver
