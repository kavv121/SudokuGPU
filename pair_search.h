template<int RSIZE>
__global__ void pair_search_v2(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;
    __shared__ uint32_t bit_counts[RSIZE*RSIZE];
    const int tb = blockIdx.x;
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){block_status = STAT_NOCHG;block_ok = 1;}
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval;
    /* Strategy: "transpose" the matrix of digits x spots, look for
       pair of digits that in total exist in 2 cells */

    //rows
    {
        const int row = tb;
        if(c == 0) {
            bit_counts[r] = 0;
        }
        __syncthreads();
        //look at bit r of cell c
        if(s.bitstate[row][c] & (1<<r)) {
            atomicOr(&bit_counts[r], (1u << c));
        }
        __syncthreads();
        if(!block_ok){goto ending;}
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
        else if(c < r) {
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
        __syncthreads();
    }
    if(!block_ok){goto ending;}
    //columns
    {
        const int col = tb;
        if(!block_ok){goto ending;;}
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
        else if(c < r) {
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
        __syncthreads();
    }

    if(!block_ok){goto ending;}
    //regions

    {
        const int regionid = tb;
        const int baser = RSIZE*(regionid/RSIZE);
        const int basec = RSIZE*(regionid%RSIZE);
        if(!block_ok){goto ending;}

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
        else if(c < r) {
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
        __syncthreads();
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
        //because we never return FINISHED, this is correct given the
        //return hierarchy
        atomicMax(rc, block_status);
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
