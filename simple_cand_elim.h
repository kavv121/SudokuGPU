template<int RSIZE>
__global__ void simple_cand_elim_v2(SudokuState<RSIZE*RSIZE> *p, int *rc) {
    __shared__ SudokuState<RSIZE*RSIZE> s;
    __shared__ int block_status, block_ok;

    //queue of (row*R + col) cells to propagate
    __shared__ int queue_data[RSIZE*RSIZE*RSIZE*RSIZE];
    __shared__ int queue_datasz;
    const int r = threadIdx.x;
    const int c = threadIdx.y;
    if(r == 0 && c == 0){
        block_status = STAT_NOCHG;
        block_ok = 1;
        queue_datasz = 0;
    }
    //copy current values
    const uint32_t myval = p->bitstate[r][c];
    s.bitstate[r][c] = myval;
    s.work_flag[r][c] = p->work_flag[r][c];
    __syncthreads();
    if(myval == 0) {
        GPU_PF("Bad tidings in (%d,%d)\n", r, c);
        block_ok = 0;
    }
    else if(myval & (myval-1)) {
        ; //do nothing
    }
    else if(s.work_flag[r][c] & SUFLG_PROPAGATE_SINGLE) {
        ; //do nothing
    }
    else {
        /* we have a singleton, put it on the shared queue */
        int oldval = atomicAdd(&queue_datasz, 1);
        queue_data[oldval] = (RSIZE*RSIZE)*r + c;
    }
    __syncthreads();
    //now we can iterate over the queue and propagate knowledge
    for(int i=0;i<queue_datasz;++i) {
        const int rr = queue_data[i]/(RSIZE*RSIZE);
        const int cc = queue_data[i]%(RSIZE*RSIZE);
        if(!(r == rr && c == cc)) {
            if((r == rr) || (c == cc) ||
               ((r/RSIZE) == (rr/RSIZE) && (c/RSIZE) == (cc/RSIZE))) {
                do_remove_mask(&s.bitstate[r][c], s.bitstate[rr][cc], &block_status);
            }
        }
    }
    __syncthreads();
    /* mark the propagation so we never do it again */
    {
        const int idx = RSIZE*RSIZE*r + c;
        const int rr = queue_data[idx]/(RSIZE*RSIZE);
        const int cc = queue_data[idx]%(RSIZE*RSIZE);
        if(idx < queue_datasz) {
            p->work_flag[rr][cc] |= SUFLG_PROPAGATE_SINGLE;
        }
    }
    __syncthreads();
    if(block_ok && block_status == STAT_UPDATED) {
        p->bitstate[r][c] = s.bitstate[r][c];
    }
    if(r == 0 && c == 0) {
        if(!block_ok) {
            block_status = STAT_NOTOK;
        }
        *rc = block_status;
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
