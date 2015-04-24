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
