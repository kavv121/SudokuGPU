#include <iostream>
#include <string>
#include <fstream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <cassert>
#include <sys/time.h>


//#define GPUDEBUG

#ifdef GPUDEBUG
#define GPU_PF(...) printf(__VA_ARGS__)
#else
#define GPU_PF(...)
#endif

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

__device__ inline void do_remove_mask(uint32_t *data, int mask, int *bstatus) {
    if(atomicAnd(data, ~mask) & mask){
        *bstatus = STAT_UPDATED;
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
        GPU_PF("Bad tidings in (%d,%d)\n", r, c);
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
            do_remove_mask(&s.bitstate[r][oc], myval, &block_status);
        }
    }
    //column
    for(int row=0;row<RSIZE*RSIZE;++row) {
        if(row != r) {
            do_remove_mask(&s.bitstate[row][c], myval, &block_status);
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
                do_remove_mask(&s.bitstate[baser+dr][basec+dc], myval, &block_status);
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

template<int RSIZE>
__global__ void pair_search(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    __shared__ uint32_t bit_counts[RSIZE*RSIZE];
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval; 
    /* Strategy: "transpose" the matrix of digits x spots, look for
       pair of digits that in total exist in 2 cells */

    //rows
    for(int row=0;row<RSIZE*RSIZE;++row) {
        bit_counts[r] = 0;
        __syncthreads();
        //look at bit r of cell c
        if(s.bitstate[row][c] & (1<<r)) {
            atomicOr(&bit_counts[r], (1u << c));
        }
        __syncthreads();
        if(!block_ok){break;}
        //check if the pair of digits (r,c) happens in exactly 2 places
        if(r < c) {
            const uint32_t x = (bit_counts[r] | bit_counts[c]);
            int ct = __popc(x);
            if(ct <= 1) {
                //There aren't enough cells for this pair!
                //That's no good!
                block_ok = 0;
            }
            else if(ct == 2) {
                //r and c are a pair!
                //we set to two cells with this pair to the bitmask
                //with bits r an c on
                const uint32_t qq = (1u<<r) | (1u<<c);
                for(int t=0;t<RSIZE*RSIZE;++t) {
                    if(x & (1u<<t)) 
                    {
                        if(qq != (qq | atomicAnd(&s.bitstate[row][t], qq))) {
                            block_status = STAT_UPDATED;
                        }
                    }
                }
            }
        }
        __syncthreads();
        //check if the pair of cells (r, c) are the same and have only 2 digits
        if(r < c) {
            const uint32_t cella = s.bitstate[row][r];
            const uint32_t cellb = s.bitstate[row][c];
            if(cella == cellb) {
                if(2 == __popc(cella)) {
                    //we found a pair of digits, apply it to everything else
                    for(int t=0;t<RSIZE*RSIZE;++t) {
                        if(t != r && t != c) {
                            do_remove_mask(&s.bitstate[row][t], cella, &block_status);
                        }
                    }
                }
            }
        }
    }
    if(!block_ok){goto ending;}
    //columns
    for(int col=0;col<RSIZE*RSIZE;++col) {
        if(!block_ok){break;}
        bit_counts[r] = 0;
        __syncthreads();
        //look at bit r of cell c
        if(s.bitstate[c][col] & (1<<r)) {
            atomicOr(&bit_counts[r], (1u << c));
        }
        __syncthreads();
        //check if the pair of digits (r,c) happens in exactly 2 places
        if(r < c) {
            const uint32_t x = (bit_counts[r] | bit_counts[c]);
            int ct = __popc(x);
            if(ct <= 1) {
                //There aren't enough cells for this pair!
                //That's no good!
                block_ok = 0;
            }
            else if(ct == 2) {
                //r and c are a pair!
                //we set to two cells with this pair to the bitmask
                //with bits r an c on
                const uint32_t qq = (1u<<r) | (1u<<c);
                for(int t=0;t<RSIZE*RSIZE;++t) {
                    if(x & (1u<<t)) 
                    {
                        if(qq != (qq|atomicAnd(&s.bitstate[t][col], qq))) {
                            block_status = STAT_UPDATED;
                        }
                    }
                }
            }
        }
        __syncthreads();
        //check if the pair of cells (r, c) are the same and have only 2 digits
        if(r < c) {
            const uint32_t cella = s.bitstate[r][col];
            const uint32_t cellb = s.bitstate[c][col];
            if(cella == cellb) {
                if(2 == __popc(cella)) {
                    //we found a pair of digits, apply it to everything else
                    for(int t=0;t<RSIZE*RSIZE;++t) {
                        if(t != r && t != c) {
                            do_remove_mask(&s.bitstate[t][col], cella, &block_status);
                        }
                    }
                }
            }
        }
    }

    if(!block_ok){goto ending;}
    //regions

    for(int regionid=0;regionid<RSIZE*RSIZE;++regionid) {
        const int baser = RSIZE*(regionid/RSIZE);
        const int basec = RSIZE*(regionid%RSIZE);
        if(!block_ok){break;}

        bit_counts[r] = 0;
        __syncthreads();
        //look at bit r of cell c
        if(s.bitstate[baser+(c/RSIZE)][basec+(c%RSIZE)] & (1<<r)) {
            atomicOr(&bit_counts[r], (1u << c));
        }
        __syncthreads();
        //check if the pair of digits (r,c) happens in exactly 2 places
        if(r < c) {
            const uint32_t x = (bit_counts[r] | bit_counts[c]);
            int ct = __popc(x);
            if(ct <= 1) {
                //There aren't enough cells for this pair!
                //That's no good!
                block_ok = 0;
            }
            else if(ct == 2) {
                //r and c are a pair!
                //we set to two cells with this pair to the bitmask
                //with bits r an c on
                const uint32_t qq = (1u<<r) | (1u<<c);
                for(int t=0;t<RSIZE*RSIZE;++t) {
                    if(x & (1u<<t)) 
                    {
                        if(qq != (qq|atomicAnd(&s.bitstate[baser+(t/RSIZE)][basec+(t%RSIZE)], qq))) {
                            block_status = STAT_UPDATED;
                        }
                    }
                }
            }
        }
        __syncthreads();
        //check if the pair of cells (r, c) are the same and have only 2 digits
        if(r < c) {
            const uint32_t cella = s.bitstate[baser+(r/RSIZE)][basec+(r%RSIZE)];
            const uint32_t cellb = s.bitstate[baser+(c/RSIZE)][basec+(c%RSIZE)];
            if(cella == cellb) {
                if(2 == __popc(cella)) {
                    //we found a pair of digits, apply it to everything else
                    for(int t=0;t<RSIZE*RSIZE;++t) {
                        if(t != r && t != c) {
                            do_remove_mask(&s.bitstate[baser+(t/RSIZE)][basec+(t%RSIZE)], cella, &block_status);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    
ending:
    if(block_ok && block_status == STAT_UPDATED)
    { //implies ok && finalval has 1 bit set
        p->bitstate[r][c] = s.bitstate[r][c];
    }
    __syncthreads();
    if(r == 0 && c == 0) {
        if(!block_ok) {
            block_status = STAT_NOTOK;
        }
        *rc = block_status;
    }
}

/* "Pointing Pairs", but also includes other (2 of box/row/column) interactions */
template<int RSIZE>
__global__ void intersection_search(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    __shared__ uint32_t bit_counts[RSIZE*RSIZE];
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval; 
    __syncthreads();
    /* look for pairs/triples in box */
    for(int boxr=0;boxr<RSIZE;++boxr)
    for(int boxc=0;boxc<RSIZE;++boxc)
    {
        const int baser = RSIZE * boxr;
        const int basec = RSIZE * boxc;

        //thread (r,c) will check for digit r in cell c of the box
        if(c == 0) {
            bit_counts[r] = 0;
        }
        __syncthreads();
        if(s.bitstate[baser+(c/RSIZE)][basec+(c%RSIZE)] & (1<<r)) {
            atomicOr(&bit_counts[r], (1u << c));
        }
        __syncthreads();

        //now we check for each digit whether it signals a row or column intersection
        //for digit d
        //if bit counts mask == xxx000000
        //  eliminate rest in row 0
        //if bit counts mask == 000xxx000
        //  eliminate rest in row 1
        //...
        //TODO: generalize this instead of hardcoding 9x9 values
        {
            //row intersection
            if(!(c >= basec && c < basec+RSIZE)) {
                for(int x=0,mymask=((1u<<RSIZE)-1);x<RSIZE;++x, mymask<<=RSIZE) {
                    if((bit_counts[r] & mymask) == bit_counts[r]) {
                        do_remove_mask(&s.bitstate[baser+x][c], (1u<<r), &block_status);
                    }
                }
            }
            //column intersection
            if(!(c >= baser && c < baser+RSIZE)) {
                //TODO: find correct mask for given RSIZE
                for(int x=0,mymask=0x49;x<RSIZE;++x, mymask<<=1) {
                    if((bit_counts[r] & mymask) == bit_counts[r]) {
                        do_remove_mask(&s.bitstate[c][basec+x], (1u<<r), &block_status);
                    }
                }
            }
        }
        __syncthreads();
    }
    /* TODO: row/box and col/box interaction */
    if(block_ok && block_status == STAT_UPDATED)
    { //implies ok && finalval has 1 bit set
        p->bitstate[r][c] = s.bitstate[r][c];
    }
    __syncthreads();
    if(r == 0 && c == 0) {
        if(!block_ok) {
            block_status = STAT_NOTOK;
        }
        *rc = block_status;
    }
}

template<int RSIZE>
__global__ void xwing_search(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    __shared__ uint32_t bit_counts[RSIZE*RSIZE];
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval; 
    __syncthreads();

    for(int dig=0;dig<RSIZE*RSIZE;++dig) {
        //look by row
        if(c == 0)
            bit_counts[r] = 0;
        __syncthreads();
        if((s.bitstate[r][c] & (1u<<dig))) {
            atomicOr(&bit_counts[r], (1u << c));
        }
        __syncthreads();
        //now we look for 2 rows that are identical and
        //have exactly two spots open
        if(r < c) {
            if(bit_counts[r] == bit_counts[c] && __popc(bit_counts[r]) == 2) {
                //we have to remove this digit from the other things in the two bits that are there
                int col_a = __ffs(bit_counts[r])-1;
                int col_b = 31-__clz(bit_counts[r]);
                //GPU_PF("Found xw %x %d by row %d %d, cols %d %d\n", bit_counts[r], dig, r, c, col_a, col_b);

                for(int t=0;t<RSIZE*RSIZE;++t) {
                    if(t != r && t != c) {
                        do_remove_mask(&s.bitstate[t][col_a], (1u << dig), &block_status);
                        do_remove_mask(&s.bitstate[t][col_b], (1u << dig), &block_status);
                    }
                }
            }
        }
        //now look by column
        if(c == 0)
            bit_counts[r] = 0;
        __syncthreads();
        if((s.bitstate[r][c] & (1u<<dig))) {
            atomicOr(&bit_counts[c], (1u << r));
        }
        __syncthreads();
        if(r < c) {
            if(bit_counts[r] == bit_counts[c] && __popc(bit_counts[r]) == 2) {
                //we have to remove this digit from the other things in the two bits that are there
                int row_a = __ffs(bit_counts[r])-1;
                int row_b = 31-__clz(bit_counts[r]);
                //GPU_PF("Found xw %x %d by col %d %d, rows %d %d\n", bit_counts[r], dig, r, c, row_a, row_b);
                for(int t=0;t<RSIZE*RSIZE;++t) {
                    if(t != r && t != c) {
                        do_remove_mask(&s.bitstate[row_a][t], (1u << dig), &block_status);
                        do_remove_mask(&s.bitstate[row_b][t], (1u << dig), &block_status);
                    }
                }
            }
        }
        __syncthreads();
    }

    if(block_ok && block_status == STAT_UPDATED)
    {
        p->bitstate[r][c] = s.bitstate[r][c];
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
int check_state(const SudokuState<RSIZE*RSIZE> &s) {
    const char *foo = getenv("VERBOSE");
    bool ok = true;
    for(int r=0;r<RSIZE*RSIZE;++r) {
        for(int c=0;c<RSIZE*RSIZE;++c) {
            const uint32_t bs = s.bitstate[r][c];
            if(bs == 0 || (bs & (bs-1)) != 0) {
                if(foo)
                    std::cerr << "Row " << r << " " << "col " << c << " not singleton!" << std::endl;
                ok = false;
            }
        }
    }
    if(!ok){return 0;}
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
            if(foo)
                std::cerr << "Row " << r << " is no good" << std::endl;
            ok = false;
            break;
        }
    }
    if(!ok){return 0;}
    for(int c=0;c<RSIZE*RSIZE;++c) {
        uint32_t xx = 0;
        for(int i=0;i<9;++i) {
            const int nr = i;
            const int nc = c;
            xx |= s.bitstate[nr][nc];
        }
        if(xx != GOAL) {
            if(foo)
                std::cerr << "Col " << c << " is no good" << std::endl;
            ok = false;
            break;
        }
    }
    if(!ok){return 0;}
    for(int br=0;br<RSIZE && ok;++br) {
        for(int bc=0;bc<RSIZE;++bc) {
            uint32_t xx = 0;
            for(int i=0;i<9;++i) {
                const int nr = RSIZE*br + (i/3);
                const int nc = RSIZE*bc + (i%3);
                xx |= s.bitstate[nr][nc];
            }
            if(xx != GOAL) {
                if(foo)
                    std::cerr << "Region " << br << "," << bc << " is no good" << std::endl;
                ok = false;
                break;
            }
        }
    }
    if(!ok){return 0;}
    if(foo)
        std::cerr << "ALL GOOD!" << std::endl;
    return 1;
}

static double timeval_diff(const struct timeval *start, const struct timeval *end) {
    return 1e-6 * (end->tv_usec - start->tv_usec) + 
                  (end->tv_sec  - start->tv_sec);
}

#define RSIZE 3
#define SIZE 9
__device__ int naive_recurse(SudokuState<9> *p) {
    //const int RSIZE = 3;
    //const int SIZE = RSIZE*RSIZE;
    uint32_t hold_row[SIZE];
    uint32_t hold_col[SIZE];
    uint32_t hold_region[SIZE];
    for(int r=0;r<SIZE;++r) {
        for(int c=0;c<SIZE;++c) {
            if(__popc(p->bitstate[r][c]) == 0){return 0;}
        }
    }
    for(int r=0;r<SIZE;++r) {
        for(int c=0;c<SIZE;++c) {
            if(__popc(p->bitstate[r][c]) > 1 ){
                uint32_t oldval = p->bitstate[r][c];
                for(int q=0;q<SIZE;++q) {
                    if(oldval & (1u<<q)) {
                        const uint32_t mask = ~(1u<<q);
                        const int baser = RSIZE * (r/RSIZE);
                        const int basec = RSIZE * (c/RSIZE);
                        //try doing this with q
                        for(int i=0;i<SIZE;++i){
                            if(i != c) {
                                hold_row[i] = p->bitstate[r][i];
                                p->bitstate[r][i] &= mask;
                            }
                            if(i != r) {
                                hold_col[i] = p->bitstate[i][c];
                                p->bitstate[i][c] &= mask;
                            }
                            const int nr = baser + (i/RSIZE);
                            const int nc = basec + (i%RSIZE);
                            if(!(nr == r && nc == c)) {
                                hold_region[i] = p->bitstate[nr][nc];
                                p->bitstate[nr][nc] &= mask;
                            }
                        }
                        p->bitstate[r][c] = (1u<<q);
                        if(naive_recurse(p)) {return 1;}
                        p->bitstate[r][c] = oldval;
                        for(int i=SIZE-1;i>=0;--i){
                            const int nr = baser + (i/RSIZE);
                            const int nc = basec + (i%RSIZE);
                            if(!(nr == r && nc == c)) {
                                p->bitstate[nr][nc] = hold_region[i];
                            }
                            if(i != r) {
                                p->bitstate[i][c] = hold_col[i];
                            }
                            if(i != c) {
                                p->bitstate[r][i] = hold_row[i];
                            }
                        }
                    }
                }
                return 0;
            }
        }
    }
    return 1;
}
int cpu_naive_recurse(SudokuState<9> *p) {
    //const int RSIZE = 3;
    //const int SIZE = RSIZE*RSIZE;
    uint32_t hold_row[SIZE];
    uint32_t hold_col[SIZE];
    uint32_t hold_region[SIZE];
    for(int r=0;r<SIZE;++r) {
        for(int c=0;c<SIZE;++c) {
            uint32_t x = p->bitstate[r][c];
            if(x == 0){return 0;}
            if(!(x & (x-1))) {
                for(int i=0;i<SIZE;++i)
                {
                    if(i != r && p->bitstate[i][c] == x){return 0;}
                    if(i != c && p->bitstate[r][i] == x){return 0;}
                    const int nr = (RSIZE*(r/RSIZE))+(i/RSIZE);
                    const int nc = (RSIZE*(c/RSIZE))+(i%RSIZE);
                    if(!(nr == r && nc == c) && p->bitstate[nr][nc] == x){return 0;}
                }
            }
        }
    }
    for(int r=0;r<SIZE;++r) {
        for(int c=0;c<SIZE;++c) {
            uint32_t oldval = p->bitstate[r][c];
            if((oldval & (oldval-1)) != 0){
                for(int q=0;q<SIZE;++q) {
                    if(oldval & (1u<<q)) {
                        const uint32_t mask = ~(1u<<q);
                        const int baser = RSIZE * (r/RSIZE);
                        const int basec = RSIZE * (c/RSIZE);
                        //try doing this with q
                        for(int i=0;i<SIZE;++i){
                            if(i != c) {
                                hold_row[i] = p->bitstate[r][i];
                                p->bitstate[r][i] &= mask;
                            }
                            if(i != r) {
                                hold_col[i] = p->bitstate[i][c];
                                p->bitstate[i][c] &= mask;
                            }
                            const int nr = baser + (i/RSIZE);
                            const int nc = basec + (i%RSIZE);
                            if(!(nr == r && nc == c)) {
                                hold_region[i] = p->bitstate[nr][nc];
                                p->bitstate[nr][nc] &= mask;
                            }
                        }
                        p->bitstate[r][c] = (1u<<q);
                        if(cpu_naive_recurse(p)) {return 1;}
                        p->bitstate[r][c] = oldval;
                        for(int i=SIZE-1;i>=0;--i){
                            const int nr = baser + (i/RSIZE);
                            const int nc = basec + (i%RSIZE);
                            if(!(nr == r && nc == c)) {
                                p->bitstate[nr][nc] = hold_region[i];
                            }
                            if(i != r) {
                                p->bitstate[i][c] = hold_col[i];
                            }
                            if(i != c) {
                                p->bitstate[r][i] = hold_row[i];
                            }
                        }
                    }
                }
                return 0;
            }
        }
    }
    return 1;
}
#undef RSIZE
#undef SIZE

__global__ void sudokusolver_gpu_naive(SudokuState<9> *p, int *rc) {
    int rrc = naive_recurse(p);
    *rc = rrc;
}


template<int RSIZE>
__global__ void sudokusolver_gpu_main(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    const dim3 num_block(1,1,1);
    const dim3 threads_per_block(9,9,1);
    for(*rc = STAT_UPDATED;*rc == STAT_UPDATED;) {
        *rc = STAT_NOCHG;
        __syncthreads();

        simple_cand_elim<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("SIMPLE - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}

        singleton_search<RSIZE><<<num_block, threads_per_block>>>(p,rc);
        cudaDeviceSynchronize();
        GPU_PF("SINGLETON - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}

        pair_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("PAIR SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}

        intersection_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("INTERSECTION SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}

        xwing_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("XWING SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}
    }
}

void test_basics2(SudokuState<9> &state) {
    SudokuState<9> *d_state;
    int *d_rc;
    int h_rc;
    /* TODO: better timing */
    struct timeval tstart, tend;
    const char *foo = getenv("CPUMODE");

    GPU_CHECKERROR(cudaMalloc((void **)&d_state, 
                              sizeof(SudokuState<9>)));
    GPU_CHECKERROR(cudaMalloc((void **)&d_rc, 
                              sizeof(int)));
    gettimeofday(&tstart, 0);
    if(foo) {
        h_rc = cpu_naive_recurse(&state);
    }
    else {
        GPU_CHECKERROR(cudaMemset(d_rc, 0, sizeof(int)));

        GPU_CHECKERROR(cudaMemcpy(d_state, &state, sizeof(SudokuState<9>), cudaMemcpyHostToDevice));
        const dim3 num_block(1,1,1);
        const dim3 threads_per_block(1,1,1);
        sudokusolver_gpu_main<3><<<num_block, threads_per_block>>>(d_state, d_rc);
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaMemcpy(&h_rc, d_rc, sizeof(int), cudaMemcpyDeviceToHost));
        GPU_CHECKERROR(cudaMemcpy(&state, d_state, sizeof(SudokuState<9>), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    gettimeofday(&tend, 0);
    std::cerr << "GOT OVERALL RC " << h_rc << std::endl;
    std::cerr << "TOOK TIME " << timeval_diff(&tstart, &tend) * 1000.0 << " ms" << std::endl;
    cudaFree(d_state);
    cudaFree(d_rc);
    if(check_state<3>(state)) {
        std::cerr << "PASS" << std::endl;
    }
    else {
        std::cerr << "*****FAIL*****" << std::endl;
    }
}

#if 0
void test_basics(SudokuState<9> &state) {
    SudokuState<9> *d_state;
    int *d_rc;
    int h_rc;
    /* TODO: better timing */
    struct timeval tstart, tend;

    gettimeofday(&tstart, 0);
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

        pair_search<3><<<num_block, threads_per_block>>>(d_state, d_rc);
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaMemcpy(&h_rc, d_rc, sizeof(int), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        fprintf(stderr, "PAIR SEARCH - GOT RC %d\n", h_rc);
        if(h_rc != STAT_NOCHG){continue;}

    }
    GPU_CHECKERROR(cudaMemcpy(&state, d_state, sizeof(SudokuState<9>), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    gettimeofday(&tend, 0);
    std::cerr << "TOOK TIME " << timeval_diff(&tstart, &tend) * 1000.0 << " ms" << std::endl;
    cudaFree(d_state);
    cudaFree(d_rc);
    

    
    //print_state(state);
    check_state<3>(state);
}
#endif


int main(int argc, char **argv) {
    if(argc > 1) {
        std::cerr << "Entering bulk mode on file " << argv[1] << std::endl;
        std::ifstream fin(argv[1]);
        std::vector<std::string> vs;
        {
            std::string s;
            while(fin >> s) {
                if(s.size() != 81){
                    std::cerr << "Warning, incomplete string '" << s << "', skipping" << std::endl;
                }
                else {
                    vs.push_back(s);
                }
            }
        }
        std::vector<SudokuState<9> > states(vs.size());
        for(int i=0;i<vs.size();++i)
        {
            SudokuProblem<9> problem;
            memset(&problem, 0, sizeof(problem));
            const std::string &s = vs[i];
            for(int t=0;t<s.size();++t) {
                if(s[t] >= '1' && s[t] <= '9') {
                    int dig = s[t] - '0';
                    int r = t/9;
                    int c = t%9;
                    problem.givens[r][c] = dig;
                }
            }
            fill_state_from_problem(&states[i], problem);
        }
        for(int i=0;i<states.size();++i) {
            test_basics2(states[i]);
        }
        return 0;
    }
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
    //print_state(mystate);
    test_basics2(mystate);
    return 0;
}
