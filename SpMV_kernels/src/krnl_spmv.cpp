#include "krnl_spmv.h"

#ifdef __VITIS_CL__
extern "C" {
#endif
void krnl_spmv(const data_t* values, const data_t *colIdx, const data_t *rowPtr,
        const data_t* x, data_t *y, uint64_t batch_size) {

    for (index_t iter = 0; iter < batch_size; iter++) {
        for (index_t i = 0; i < NN; i++) {
            data_t yt = 0;
            for (index_t k = ARRAY2(rowPtr,iter,i,NN+1); k < ARRAY2(rowPtr,iter,i+1,NN+1); k++) {
                yt += ARRAY2(values,iter,k,NNZ) *
                    ARRAY2(x,iter,(ARRAY2(colIdx,iter,k,NNZ)),MM);
            }
            ARRAY2(y,iter,i,NN) = yt;
        }
    }
}

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif
