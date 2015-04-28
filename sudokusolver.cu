#include <iostream>
#include <string>
#include <fstream>
#include <fstream>
#include <vector>
#include <stdint.h>
#include <cassert>
#include <sys/time.h>


#define NUM_STACK 80
//#define GPUDEBUG
#define ENABLE_PAIR
//#define ENABLE_TRIPLE
//#define ENABLE_INTERSECTION
//#define ENABLE_XWING
//#define ENABLE_YWING

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
    STAT_NOTOK = 2,
    STAT_FINISHED = 3,
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
    int8_t curr_r, curr_c, curr_dig; //counter for recursion
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


#include <quickcheck.h>
#include <simple_cand_elim.h>
#include <singleton_search.h>
#include <pair_search.h>


template<int RSIZE>
__global__ void triple_search(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    __shared__ uint32_t bit_counts[RSIZE*RSIZE];
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval;

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

        //check if the pair of digits (r,c) has a third digit so that they happen in exactly 3 places
        if(r < c) {
            const uint32_t xx = (bit_counts[r] | bit_counts[c]);
            for(int odig=c+1;odig<RSIZE*RSIZE;++odig) {
                const uint32_t x = (bit_counts[odig] | xx);
                int ct = __popc(x);
                if(ct <= 2) {
                    //There aren't enough cells
                    //That's no good!
                    block_ok = 0;
                }
                else if(ct == 3) {
                    const uint32_t qq = (1u<<r)|(1u<<c)|(1u<<odig);
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
        }
        __syncthreads();
        //check if the pair of cells (r, c) has a third cell that has only 3 digits
        if(r < c) {
            const uint32_t basepair = (
                    s.bitstate[row][r]|
                    s.bitstate[row][c]);
            if(__popc(basepair) <= 3) {
                for(int ocell=c+1;ocell<RSIZE*RSIZE;++ocell) {
                    const uint32_t tmask = (basepair|
                             s.bitstate[row][ocell]);
                    if(__popc(tmask) == 3) {
                        for(int t=0;t<RSIZE*RSIZE;++t) {
                            if(t != r && t != c && t != ocell) {
                                do_remove_mask(&s.bitstate[row][t],
                                               tmask, &block_status);
                            }
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

        //check if the pair of digits (r,c) has a third digit so that they happen in exactly 3 places
        if(r < c) {
            const uint32_t xx = (bit_counts[r] | bit_counts[c]);
            for(int odig=c+1;odig<RSIZE*RSIZE;++odig) {
                const uint32_t x = (bit_counts[odig] | xx);
                int ct = __popc(x);
                if(ct <= 2) {
                    //There aren't enough cells
                    //That's no good!
                    block_ok = 0;
                }
                else if(ct == 3) {
                    const uint32_t qq = (1u<<r)|(1u<<c)|(1u<<odig);
                    for(int t=0;t<RSIZE*RSIZE;++t) {
                        if(x & (1u<<t))
                        {
                            if(qq != (qq | atomicAnd(&s.bitstate[t][col], qq))) {
                                block_status = STAT_UPDATED;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        //check if the pair of cells (r, c) has a third cell that has only 3 digits
        if(r < c) {
            const uint32_t basepair = (
                    s.bitstate[r][col]|
                    s.bitstate[c][col]);
            if(__popc(basepair) <= 3) {
                for(int ocell=c+1;ocell<RSIZE*RSIZE;++ocell) {
                    const uint32_t tmask = (basepair|
                             s.bitstate[ocell][col]);
                    if(__popc(tmask) == 3) {
                        for(int t=0;t<RSIZE*RSIZE;++t) {
                            if(t != r && t != c && t != ocell) {
                                do_remove_mask(&s.bitstate[t][col],
                                               tmask, &block_status);
                            }
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

        //check if the pair of digits (r,c) has a third digit so that they happen in exactly 3 places
        if(r < c) {
            const uint32_t xx = (bit_counts[r] | bit_counts[c]);
            for(int odig=c+1;odig<RSIZE*RSIZE;++odig) {
                const uint32_t x = (bit_counts[odig] | xx);
                int ct = __popc(x);
                if(ct <= 2) {
                    //There aren't enough cells
                    //That's no good!
                    block_ok = 0;
                }
                else if(ct == 3) {
                    const uint32_t qq = (1u<<r)|(1u<<c)|(1u<<odig);
                    for(int t=0;t<RSIZE*RSIZE;++t) {
                        if(x & (1u<<t))
                        {
                            if(qq != (qq | atomicAnd(&s.bitstate[baser+(t/RSIZE)][basec+(t%RSIZE)], qq))) {
                                block_status = STAT_UPDATED;
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        //check if the pair of cells (r, c) has a third cell that has only 3 digits
        if(r < c) {
            const uint32_t basepair = (
                    s.bitstate[baser+(r/RSIZE)][basec+(r%RSIZE)]|
                    s.bitstate[baser+(c/RSIZE)][basec+(c%RSIZE)]);
            if(__popc(basepair) <= 3) {
                for(int ocell=c+1;ocell<RSIZE*RSIZE;++ocell) {
                    const uint32_t tmask = (basepair|
                             s.bitstate[baser+(ocell/RSIZE)][basec+(ocell%RSIZE)]);
                    if(__popc(tmask) == 3) {
                        for(int t=0;t<RSIZE*RSIZE;++t) {
                            if(t != r && t != c && t != ocell) {
                                do_remove_mask(&s.bitstate[baser+(t/RSIZE)][basec+(t%RSIZE)],
                                               tmask, &block_status);
                            }
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

ending:
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

template<int RSIZE>
__global__ void ywing_search(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    //left versus right
    __shared__ uint32_t bit_counts[RSIZE*RSIZE][RSIZE*RSIZE][2];
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    const int mybits = __popc(myval);
    s.bitstate[r][c] = myval;
    __syncthreads();

    //iterate over every cell with exactly 2 values,
    //propagate a search for other bivalued cells with one value in common,
    //set bitmask for the other side to indicate that cell
    //then take the set bitmasks and propagate those, hopefully you then have overlap to
    //remove from main board

    for(int keyr=0;keyr<RSIZE*RSIZE;++keyr) {
        for(int keyc=0;keyc<RSIZE*RSIZE;++keyc) {
            const uint32_t kval = s.bitstate[keyr][keyc];
            const uint32_t lowval = __ffs(kval)-1;
            if(__popc(kval) != 2){continue;}
            bit_counts[r][c][0] = bit_counts[r][c][1] = 0;
            __syncthreads();
            //now attempt to propagate
            if((r == keyr) || (c == keyc) || ((r/RSIZE) == (keyr/RSIZE) && (c/RSIZE) == (keyc/RSIZE))) {
                if(mybits == 2) {
                    uint32_t x = (myval & (~kval));
                    if(x && !(x & (x-1))) {
                        int target_idx = 0;
                        if(myval == (x | (1u<<lowval))) {
                            target_idx = 0;
                        }
                        else {
                            target_idx = 1;
                        }
                        //propagate to all cells stemming from this one
                        for(int t=0;t<9;++t) {
                            if(t != c){
                                atomicOr(&bit_counts[r][t][target_idx], x);
                            }
                            if(t != r) {
                                atomicOr(&bit_counts[t][c][target_idx], x);
                            }
                            if(t != (RSIZE*(r%RSIZE) + (c%RSIZE))) {
                                atomicOr(&bit_counts[RSIZE*(r/RSIZE)+(t/3)][RSIZE*(c/RSIZE)+(t%3)][target_idx], x);
                            }
                        }
                    }
                }
            }
            __syncthreads();
            //now do intersection on each propagated thing
            {
                const uint32_t rval = (bit_counts[r][c][0]&bit_counts[r][c][1]);
                if(rval) {
                    do_remove_mask(&s.bitstate[r][c], rval, &block_status);
                }
            }
            __syncthreads();
            if(block_status == STAT_UPDATED) {
                goto ending;
            }
            if(!block_ok) {
                goto ending;
            }
        }
    }
ending:;
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
        for(int i=0;i<RSIZE*RSIZE;++i) {
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
        for(int i=0;i<RSIZE*RSIZE;++i) {
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
            for(int i=0;i<RSIZE*RSIZE;++i) {
                const int nr = RSIZE*br + (i/RSIZE);
                const int nc = RSIZE*bc + (i%RSIZE);
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

#define SIZE 9
#define RSIZE 3

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
#undef SIZE
#undef RSIZE

template<int RSIZE>
__device__ bool iterate_guess(SudokuState<RSIZE*RSIZE> *guesser, SudokuState<RSIZE*RSIZE> *dest) {
    int rr = guesser->curr_r;
    int cc = guesser->curr_c;
    int dd = guesser->curr_dig;
    //increment for next thing
    ++dd;
    if(dd == RSIZE*RSIZE){
        dd = 0; ++cc;
        if(cc == RSIZE*RSIZE) {
            cc = 0;++rr;
        }
    }
    for(;rr < RSIZE*RSIZE;++rr) {
        for(;cc < RSIZE*RSIZE;++cc) {
            if(__popc(guesser->bitstate[rr][cc]) <= 1){dd = 0;continue;}
            for(;dd < RSIZE*RSIZE;++dd) {
                if(guesser->bitstate[rr][cc] & (1u<<dd)) {
                    //we found a spot to try
                    memcpy(dest, guesser, sizeof(SudokuState<RSIZE*RSIZE>));
                    dest->bitstate[rr][cc] = (1u << dd);
                    guesser->curr_r = rr;
                    guesser->curr_c = cc;
                    guesser->curr_dig = dd;
                    return true;
                }
            }
            dd = 0;
        }
        cc = 0;
    }
    return false;
}


template<int RSIZE>
__global__ void sudokusolver_gpu_main(SudokuState<RSIZE*RSIZE> *p, SudokuState<RSIZE*RSIZE> *save_stack, int ss_size, int *rc) {
    const dim3 num_block(1,1,1);
    const dim3 super_num_block(RSIZE*RSIZE,1,1);
    const dim3 threads_per_block(RSIZE*RSIZE,RSIZE*RSIZE,1);
    int stack_ptr = 0;
thetop:;
    for(*rc = STAT_UPDATED;*rc == STAT_UPDATED;) {
        *rc = STAT_NOCHG;
        __syncthreads();

        quickcheck<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("QUICKCHECK - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}

        //simple_cand_elim<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        simple_cand_elim_v2<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("SIMPLE - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}

        singleton_search<RSIZE><<<num_block, threads_per_block>>>(p,rc);
        //singleton_search_v2<RSIZE><<<num_block, threads_per_block>>>(p,rc);
        cudaDeviceSynchronize();
        GPU_PF("SINGLETON - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}

#ifdef ENABLE_PAIR
        //pair_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        pair_search_v2<RSIZE><<<super_num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("PAIR SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}
#endif

#ifdef ENABLE_TRIPLE
        triple_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("TRIPLE SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}
#endif

#ifdef ENABLE_INTERSECTION
        intersection_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("INTERSECTION SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}
#endif

#ifdef ENABLE_XWING
        xwing_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("XWING SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}
#endif

#ifdef ENABLE_YWING
        ywing_search<RSIZE><<<num_block, threads_per_block>>>(p, rc);
        cudaDeviceSynchronize();
        GPU_PF("YWING SEARCH - GOT RC %d\n", *rc);
        if(*rc != STAT_NOCHG){continue;}
#endif

    }
    //did we win?
    if(*rc == STAT_FINISHED) {
        return;
    }
    else if(*rc == STAT_NOTOK || stack_ptr >= ss_size) {
        //we've hit a contradiction, so we need to iter the stack
        while(stack_ptr > 0) {
            /* remove the current guess from the save stack
               */
            SudokuState<RSIZE*RSIZE> &ss = save_stack[stack_ptr-1];
            int nextdig;
            for(nextdig=ss.curr_dig+1;nextdig<RSIZE*RSIZE;++nextdig) {
                if(ss.bitstate[ss.curr_r][ss.curr_c] & (1u<<nextdig)) {
                    ss.curr_dig = nextdig;
                    memcpy(p, &ss, sizeof(SudokuState<RSIZE*RSIZE>));
                    p->bitstate[ss.curr_r][ss.curr_c] = (1u<<nextdig);
                    GPU_PF("NOW TRYING %d %d %d %d\n", stack_ptr-1, ss.curr_r, ss.curr_c, nextdig);
                    goto thetop;
                }
            }
            //no next digit, we're also a failure!
            GPU_PF("Now popping to %d\n", stack_ptr-1);
            --stack_ptr;
        }
    }
    else if(stack_ptr < ss_size) {
        //we are stuck, but not dead, make a guess and add to stack, unless we're out of stack
        memcpy(&save_stack[stack_ptr], p, sizeof(SudokuState<RSIZE*RSIZE>));
        int bestr = 0;int bestc = 0; int bestsc = RSIZE*RSIZE+1;
        for(int rr=0;rr<RSIZE*RSIZE;++rr) {
            for(int cc=0;cc<RSIZE*RSIZE;++cc) {
                int x = __popc(p->bitstate[rr][cc]);
                if(x > 1 && x < bestsc) {
                    bestr = rr;bestc = cc;bestsc = x;
                }
            }
        }
        save_stack[stack_ptr].curr_r = bestr;
        save_stack[stack_ptr].curr_c = bestc;
        save_stack[stack_ptr].curr_dig = 0;
        SudokuState<RSIZE*RSIZE> &ss = save_stack[stack_ptr];
        for(int nextdig=0;nextdig<RSIZE*RSIZE;++nextdig) {
            if(ss.bitstate[ss.curr_r][ss.curr_c] & (1u<<nextdig)) {
                GPU_PF("GUESS TIME %d %d %d %d %d\n", stack_ptr, ss.curr_r, ss.curr_c, nextdig, bestsc);
                ss.curr_dig = nextdig;
                memcpy(p, &ss, sizeof(SudokuState<RSIZE*RSIZE>));
                p->bitstate[ss.curr_r][ss.curr_c] = (1u<<nextdig);
                ++stack_ptr;
                goto thetop;
            }
        }
        //we really shouldn't get here, let the whole thing die
        GPU_PF("SHOULDNT GET HERE!!!!\n");
    }
}


static SudokuState<9> *d_state;
static SudokuState<9> *d_sstack;
static int *d_rc;
static int gpumalloc = 0;

int test_basics2(SudokuState<9> &state) {

    int h_rc;
    /* TODO: better timing */
    struct timeval tstart, tend;
    const int num_stack = NUM_STACK;

    if(!gpumalloc) {
        GPU_CHECKERROR(cudaMalloc((void **)&d_state,
                                  sizeof(SudokuState<9>)));
        GPU_CHECKERROR(cudaMalloc((void **)&d_sstack,
                                  num_stack*sizeof(SudokuState<9>)));
        GPU_CHECKERROR(cudaMalloc((void **)&d_rc,
                                  sizeof(int)));
        gpumalloc = 1;
    }
    gettimeofday(&tstart, 0);
    {
        GPU_CHECKERROR(cudaMemset(d_rc, 0, sizeof(int)));

        GPU_CHECKERROR(cudaMemcpy(d_state, &state, sizeof(SudokuState<9>), cudaMemcpyHostToDevice));
        const dim3 num_block(1,1,1);
        const dim3 threads_per_block(1,1,1);
        sudokusolver_gpu_main<3><<<num_block, threads_per_block>>>(d_state, d_sstack, num_stack, d_rc);
        GPU_CHECKERROR(cudaGetLastError());
        GPU_CHECKERROR(cudaMemcpy(&h_rc, d_rc, sizeof(int), cudaMemcpyDeviceToHost));
        GPU_CHECKERROR(cudaMemcpy(&state, d_state, sizeof(SudokuState<9>), cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
    }
    gettimeofday(&tend, 0);
    std::cerr << "GOT OVERALL RC " << h_rc << std::endl;
    std::cerr << "TOOK TIME " << timeval_diff(&tstart, &tend) * 1000.0 << " ms" << std::endl;
    /* we don't free here in the interest of bulk mode */
    //cudaFree(d_state);
    //cudaFree(d_rc);
    if(check_state<3>(state)) {
        std::cerr << "PASS" << std::endl;
        return 1;
    }
    else {
        std::cerr << "*****FAIL*****" << std::endl;
        return 0;
    }
}

template<int RSIZE>
void run_batch(SudokuState<RSIZE*RSIZE> *states, size_t num_states) {
    const int num_streams = 8;
    SudokuState<RSIZE*RSIZE> *d_states[num_streams];
    SudokuState<RSIZE*RSIZE> *d_sstack[num_streams];
    int *d_rcs;
    const int num_stack = NUM_STACK;
    int VV = 0;
    if(getenv("VERBOSE") != NULL){VV = 1;}

    struct timeval tstart, tend;
    std::vector<int> h_rcs(num_states, 0);
    std::vector<cudaEvent_t> e_starts(num_states);
    std::vector<cudaEvent_t> e_stops(num_states);
    std::vector<cudaStream_t> streams(num_streams);

    const dim3 num_block(1,1,1);
    const dim3 threads_per_block(1,1,1);
    for(int i=0;i<num_streams;++i) {
        GPU_CHECKERROR(cudaMalloc((void **)&d_states[i],
                                  sizeof(states[0])));
        GPU_CHECKERROR(cudaMalloc((void **)&d_sstack[i],
                                  num_stack*sizeof(states[0])));
        GPU_CHECKERROR(cudaStreamCreate(&streams[i]));

    }
    for(int i=0;i<num_states;++i) {
        GPU_CHECKERROR(cudaEventCreate(&e_starts[i]));
        GPU_CHECKERROR(cudaEventCreate(&e_stops[i]));
    }
    GPU_CHECKERROR(cudaMalloc((void **)&d_rcs,
                              num_states * sizeof(int)));

    gettimeofday(&tstart, 0);
    {
        GPU_CHECKERROR(cudaMemset((void *)d_rcs, 0, num_states * sizeof(int)));
        for(int i=0;i<num_states;++i) {
            int stream_num = i % num_streams;
            GPU_CHECKERROR(cudaEventRecord(e_starts[i], streams[stream_num]));
            GPU_CHECKERROR(cudaMemcpyAsync(d_states[stream_num], states + i, sizeof(states[i]), cudaMemcpyHostToDevice, streams[stream_num]));
            sudokusolver_gpu_main<RSIZE><<<num_block, threads_per_block, 0, streams[stream_num]>>>(d_states[stream_num], d_sstack[stream_num], num_stack, &d_rcs[i]);
            GPU_CHECKERROR(cudaMemcpyAsync(states+i, d_states[stream_num], sizeof(states[i]), cudaMemcpyDeviceToHost, streams[stream_num]));
            GPU_CHECKERROR(cudaMemcpyAsync(&h_rcs[i], d_rcs + i, sizeof(int), cudaMemcpyDeviceToHost, streams[stream_num]));
            
            GPU_CHECKERROR(cudaEventRecord(e_stops[i], streams[stream_num]));
        }
    }
    for(int i=0;i<num_states;++i) {
        cudaEventSynchronize(e_stops[i]);
    }
    gettimeofday(&tend, 0);

    for(int i=0;i<num_states;++i) {
        float time_taken;
        cudaEventElapsedTime(&time_taken, e_starts[i], e_stops[i]);
        std::cout << "PUZZLE " << i << " RC " << h_rcs[i] << " TIME " << time_taken << std::endl;
        if(VV) {
            print_state<RSIZE*RSIZE>(states[i]);
        }
        if(check_state<RSIZE>(states[i])) {
            std::cout << "PASS" << std::endl;
        }
        else {
            std::cout << "FAIL" << std::endl;
        }

    }
}


int main(int argc, char **argv) {
    if(argc > 1) {
        std::cerr << "Entering bulk mode on file " << argv[1] << std::endl;
        std::ifstream fin(argv[1]);
        std::vector<std::string> vs;

        int RS = 3;
        if(argc > 2) {
            sscanf(argv[2], "%d", &RS);
        }
        if(!(RS >= 3 && RS <= 5)) {
            std::cerr << "NOT VALID SIZE" << std::endl;
        }
        int SS = RS*RS;

        {
            std::string s;
            while(fin >> s) {
                if(s.size() != SS*SS){
                    std::cerr << "Warning, incomplete string '" << s << "', skipping" << std::endl;
                }
                else {
                    vs.push_back(s);
                }
            }
        }

        if(RS == 3) {
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
            run_batch<3>(&states[0], states.size());
        }
        else if(RS == 4) {
            std::vector<SudokuState<16> > states(vs.size());
            for(int i=0;i<vs.size();++i)
            {
                SudokuProblem<16> problem;
                memset(&problem, 0, sizeof(problem));
                const std::string &s = vs[i];
                for(int t=0;t<s.size();++t) {
                    if(s[t] >= '1' && s[t] <= '9') {
                        int dig = s[t] - '0';
                        int r = t/16;
                        int c = t%16;
                        problem.givens[r][c] = dig;
                    }
                    else if(s[t] >= 'A' && s[t] <= 'G') {
                        int dig = s[t] - 'A' + 10;
                        int r = t/16;
                        int c = t%16;
                        problem.givens[r][c] = dig;
                    }
                }
                fill_state_from_problem(&states[i], problem);
            }
            run_batch<4>(&states[0], states.size());
        }
        else if(RS == 5) {
            std::vector<SudokuState<25> > states(vs.size());
            for(int i=0;i<vs.size();++i)
            {
                SudokuProblem<25> problem;
                memset(&problem, 0, sizeof(problem));
                const std::string &s = vs[i];
                for(int t=0;t<s.size();++t) {
                    if(s[t] >= '1' && s[t] <= '9') {
                        int dig = s[t] - '0';
                        int r = t/25;
                        int c = t%25;
                        problem.givens[r][c] = dig;
                    }
                    else if(s[t] >= 'A' && s[t] <= 'Z') {
                        int dig = s[t] - 'A' + 10;
                        int r = t/25;
                        int c = t%25;
                        problem.givens[r][c] = dig;

                    }
                }
                fill_state_from_problem(&states[i], problem);
            }
            run_batch<5>(&states[0], states.size());
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
    if(gpumalloc) {
        cudaFree(d_state);
        cudaFree(d_rc);
    }
    return 0;
}
