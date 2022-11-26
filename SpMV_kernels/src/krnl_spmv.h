#pragma once

#include "../../SpMV/src/util643.h"
#include "../../SpMV/src/instance643.h"

typedef uint32_t index_t;

#ifdef __VITIS_CL__
extern "C"
#endif
void krnl_spmv(const data_t *values, const data_t *colIdx, const data_t *rowPtr,
              const data_t *x, data_t *y, uint64_t batch_size);

void krnl_spmv_fast(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                    const data_t *x, data_t *y, uint64_t batch_size);

void krnl_spmv_reduced(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                       const data_t *x, data_t *y, uint64_t batch_size);

void krnl_spmv_fast_V2(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                       const data_t *x, data_t *y, uint64_t batch_size);