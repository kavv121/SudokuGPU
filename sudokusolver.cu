#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <cassert>


#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
enum {
    STAT_NOCHG = 0,
    STAT_UPDATED = 1,
    STAT_NOTOK = 2
};

enum {
    SUFLG_PROPAGATE_SINGLE = (1<<0), //propagated the fact that I solved this cell already
};

template<int SIZE>
struct SudokuProblem {
    uint32_t givens[SIZE][SIZE]; //0 if unknown, digit otherwise
};

/* warning: will break on sizes > 32 */
template<int SIZE>
struct SudokuState {
    uint32_t bitstate[SIZE][SIZE];
    uint8_t  work_flag[SIZE][SIZE]; //flags to cache work done
};

template<int SIZE>
void print_state(const SudokuState<SIZE> &s) {
    for(int r=0;r<SIZE;++r) {
        for(int c=0;c<SIZE;++c) {
            fprintf(stderr, "(%d, %d)", r,c);
            for(int t=0;t<SIZE;++t) {
                if(s.bitstate[r][c] & (1u<<t)) {
                    fprintf(stderr, " %d", t+1);
                }
            }
            fprintf(stderr, "\n");
        }
    }
}

template<int RSIZE>
__global__ void simple_cand_elim(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval; 
    s.work_flag[r][c] = p->work_flag[r][c];
    __syncthreads();
    if(myval == 0) {
        block_ok = 0;
        goto ending;
    }
    if(myval & (myval-1)) {
        goto ending;
    }
    if(s.work_flag[r][c] & SUFLG_PROPAGATE_SINGLE) {goto ending;}
    //here we have a singleton!!, atomically update neighbors
    //row
    for(int oc=0;oc<RSIZE*RSIZE;++oc) {
        if(oc != c) {
            uint32_t old = atomicAnd(&s.bitstate[r][oc], ~myval);
            if((old & (~myval)) != old) {
                block_status = STAT_UPDATED;
            }
        }
    }
    //column
    for(int row=0;row<RSIZE*RSIZE;++row) {
        if(row != r) {
            uint32_t old = atomicAnd(&s.bitstate[row][c], ~myval);
            if((old & (~myval)) != old) {
                block_status = STAT_UPDATED;
            }
        }
    }
    //block
    {
        int baser = RSIZE*(r/RSIZE);
        int basec = RSIZE*(c/RSIZE);
        for(int dr=0;dr<RSIZE;++dr) {
            for(int dc=0;dc<RSIZE;++dc) {
                if(baser + dr == r && basec + dc == c) {
                    continue;
                }
                uint32_t old = atomicAnd(&s.bitstate[baser+dr][basec+dc], ~myval);
                if((old & (~myval)) != old) {
                    block_status = STAT_UPDATED;
                }
            }
        }
    }
    p->work_flag[r][c] |= SUFLG_PROPAGATE_SINGLE;
ending:;
    __syncthreads();
    p->bitstate[r][c] = s.bitstate[r][c];
    if(r == 0 && c == 0) {
        if(!block_ok) {
            block_status = STAT_NOTOK;
        }
        *rc = block_status;
    }
}

template<int RSIZE>
__global__ void singleton_search(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval; 
    __syncthreads();


    
    uint32_t finalval = 0;
    bool ok = true;
    //don't bother if this is already a singleton or nothing
    if(myval == 0) {
        ok = false;
        goto ending;
    }
    if(!(myval & (myval-1))) {
        goto ending;
    }
    //we remove bits that are in other things
    //check row
    {
        uint32_t tval = myval;
        for(int cc=0;cc<RSIZE*RSIZE && tval;++cc) {
            if(cc == c){continue;}
            tval &= ~s.bitstate[r][cc];
        }
        if(tval) {
            if(finalval == 0 || finalval == tval) {
                finalval = tval;
            }
            else {
                ok = false;
                goto ending;
            }
        }
    }
    //column
    {
        uint32_t tval = myval;
        for(int rr=0;rr<RSIZE*RSIZE && tval;++rr) {
            if(rr == r){continue;}
            tval &= ~s.bitstate[rr][c];
        }
        if(tval) {
            if(finalval == 0 || finalval == tval) {
                finalval = tval;
            }
            else {
                ok = false;
                goto ending;
            }
        }
    }
    //region
    {
        uint32_t tval = myval;
        int base_r = RSIZE*(r/RSIZE);
        int base_c = RSIZE*(c/RSIZE);
        for(int dr=0;dr<RSIZE;++dr) {
            for(int dc=0;dc<RSIZE;++dc) {
                const int nr = base_r + dr;
                const int nc = base_c + dc;
                if(nr == r && nc == c){continue;}
                tval &= ~s.bitstate[nr][nc];

            }
        }
        if(tval) {
            if(finalval == 0 || finalval == tval) {
                finalval = tval;
            }
            else {
                ok = false;
                goto ending;
            }
        }
    }
ending:
    //either we broke something, or we concluded that two values need to fit?
    if(!ok || (finalval != 0 && (finalval & (finalval-1)) != 0)) {
        //we can do this since the change is in one direction,
        //and we sync before reading it
        block_ok = 0;
    }
    else if(finalval != 0) { //implies ok && finalval has 1 bit set
        p->bitstate[r][c] = finalval;
        //we can do this since the change is in one direction,
        //and we sync before reading it
        block_status = STAT_UPDATED;
    }
    __syncthreads();
    if(r == 0 && c == 0) {
        if(!block_ok) {
            block_status = STAT_NOTOK;
        }
        *rc = block_status;
    }
}

template<int SIZE>
void fill_state_from_problem(SudokuState<SIZE> *state, const SudokuProblem<SIZE> &problem) {
    memset(state, 0, sizeof(SudokuState<SIZE>));
    for(int r=0;r<SIZE;++r) {
        for(int c=0;c<SIZE;++c) {
            if(problem.givens[r][c] != 0) {
                assert(problem.givens[r][c] >= 1);
                assert(problem.givens[r][c] <= SIZE);
                state->bitstate[r][c] = (1u << (problem.givens[r][c]-1));
            }
            else {
                state->bitstate[r][c] = (1u << SIZE) - 1;
            }
        }
    }
}

template<int RSIZE>
void check_state(const SudokuState<RSIZE*RSIZE> &s) {
    bool ok = true;
    for(int r=0;r<RSIZE*RSIZE;++r) {
        for(int c=0;c<RSIZE*RSIZE;++c) {
            const uint32_t bs = s.bitstate[r][c];
            if(bs == 0 || (bs & (bs-1)) != 0) {
                std::cerr << "Row " << r << " " << "col " << c << " not singleton!" << std::endl;
                ok = false;
            }
        }
    }
    if(!ok){return;}
    //row check
    const uint32_t GOAL = (1u << (RSIZE*RSIZE)) - 1;
    for(int r=0;r<RSIZE*RSIZE;++r) {
        uint32_t xx = 0;
        for(int i=0;i<9;++i) {
            const int nr = r;
            const int nc = i;
            xx |= s.bitstate[nr][nc];
        }
        if(xx != GOAL) {
            std::cerr << "Row " << r << " is no good" << std::endl;
            ok = false;
            break;
        }
    }
    if(!ok){return;}
    for(int c=0;c<RSIZE*RSIZE;++c) {
        uint32_t xx = 0;
        for(int i=0;i<9;++i) {
            const int nr = i;
            const int nc = c;
            xx |= s.bitstate[nr][nc];
        }
        if(xx != GOAL) {
            std::cerr << "Col " << c << " is no good" << std::endl;
            ok = false;
            break;
        }
    }
    if(!ok){return;}
    for(int br=0;br<RSIZE && ok;++br) {
        for(int bc=0;bc<RSIZE;++bc) {
            uint32_t xx = 0;
            for(int i=0;i<9;++i) {
                const int nr = RSIZE*br + (i/3);
                const int nc = RSIZE*bc + (i%3);
                xx |= s.bitstate[nr][nc];
            }
            if(xx != GOAL) {
                std::cerr << "Region " << br << "," << bc << " is no good" << std::endl;
                ok = false;
                break;
            }
        }
    }
    if(!ok){return;}
    std::cerr << "ALL GOOD!" << std::endl;
}

void test_basics(SudokuState<9> &state) {
    SudokuState<9> *d_state;
    int *d_rc;
    int h_rc;
    GPU_CHECKERROR(cudaMalloc((void **)&d_state, 
                              sizeof(SudokuState<9>)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_rc, 
                              sizeof(int)));
    GPU_CHECKERROR(cudaMemset(d_rc, 0, sizeof(int)));

    GPU_CHECKERROR(cudaMemcpy(d_state, &state, sizeof(SudokuState<9>), cudaMemcpyHostToDevice));
    dim3 num_block(1,1,1);
    dim3 threads_per_block(9,9,1);
    for(h_rc = STAT_UPDATED;h_rc == STAT_UPDATED;)
    {
        h_rc = STAT_NOCHG;
        GPU_CHECKERROR(cudaMemset(d_rc, 0, sizeof(int)));
        //singleton_search<3><<<num_block, threads_per_block>>>(d_state, d_rc);
        simple_cand_elim<3><<<num_block, threads_per_block>>>(d_state, d_rc);
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaMemcpy(&h_rc, d_rc, sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        fprintf(stderr, "SIMPLE - GOT RC %d\n", h_rc);
        if(h_rc != STAT_NOCHG){continue;}

        singleton_search<3><<<num_block, threads_per_block>>>(d_state, d_rc);
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaMemcpy(&h_rc, d_rc, sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        fprintf(stderr, "SINGLETON - GOT RC %d\n", h_rc);
        if(h_rc != STAT_NOCHG){continue;}

    }
    GPU_CHECKERROR(cudaMemcpy(&state, d_state, sizeof(SudokuState<9>), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cudaFree(d_state);
    cudaFree(d_rc);
    

    
    //print_state(state);
    check_state<3>(state);
}


int main(int argc, char **argv) {
    std::string s;
    std::cin >> s;
    if(s.size() != 81) {
        std::cerr << "NEED 81 cells" << std::endl;
    }
    SudokuProblem<9> problem;
    memset(&problem, 0, sizeof(problem));
    for(int i=0;i<s.size();++i)
    {
        if(s[i] >= '1' && s[i] <= '9') {
            int dig = s[i] - '0';
            int r = i/9;
            int c = i % 9;
            problem.givens[r][c] = dig;
        }
    }
    SudokuState<9> mystate;
    fill_state_from_problem(&mystate, problem);
    print_state(mystate);
    test_basics(mystate);
    return 0;
}
