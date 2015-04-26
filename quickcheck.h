template<int RSIZE>
__global__ void quickcheck(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ int block_status, block_ok;
    __shared__ int nogood;
    __shared__ char tmp_block[RSIZE*RSIZE][RSIZE*RSIZE];
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;nogood=0;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    const int mydig = __ffs(myval)-1;
    __syncthreads();
    if(myval == 0) {
        GPU_PF("Bad tidings in (%d,%d)\n", r, c);
        block_ok = 0;
        nogood = 1;
    }
    else if(myval&(myval-1)) {
        nogood=1;
    }

    __syncthreads();
    if(nogood){goto ending;}
    //now everything is a singleton, so we can check every constraint and we will get yes/no
    //rows
    tmp_block[r][c] = 0;
    __syncthreads();
    tmp_block[r][mydig] = 1;
    __syncthreads();
    if(tmp_block[r][c] != 1) {
        block_ok = 0;
    }
    //cols
    tmp_block[c][mydig] = 2;
    __syncthreads();
    if(tmp_block[r][c] != 2) {
        block_ok = 0;
    }
    //regions
    tmp_block[RSIZE*(r/RSIZE) + (c/RSIZE)][mydig] = 3;
    __syncthreads();
    if(tmp_block[r][c] != 3) {
        block_ok = 0;
    }

ending:;
    __syncthreads();
    if(r == 0 && c == 0) {
        if(!block_ok) {
            block_status = STAT_NOTOK;
        }
        else if(!nogood) {
            block_status = STAT_FINISHED;
        }
        *rc = block_status;
    }
}
