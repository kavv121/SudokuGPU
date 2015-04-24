NVCC := /usr/local/cuda/bin/nvcc
NVCC_OPTS := -arch=sm_35 -rdc=true -O2
HEADERS := quickcheck.h simple_cand_elim.h singleton_search.h pair_search.h
sudokusolver: sudokusolver.cu $(HEADERS)
	$(NVCC) $(NVCC_OPTS) -I . sudokusolver.cu -lcudadevrt -o sudokusolver
