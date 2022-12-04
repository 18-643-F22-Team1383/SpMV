#pragma once

#include "../../SpMV/src/util643.h"
#include "../../SpMV/src/instance643.h"

typedef uint32_t index_t;

#ifdef __VITIS_CL__
extern "C"
{
#endif
  void krnl_spmv(const data_t *values, const data_t *colIdx, const data_t *rowPtr,
                 const data_t *x, data_t *y, uint64_t batch_size);

  void
  krnl_spmv_fast(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                 const data_t *x, data_t *y, uint64_t batch_size);

  void
  krnl_spmv_reduced(const data_t *values, const data_t *indices,
                    const data_t *x, data_t *y, uint64_t batch_size);

  void krnl_spmv_reduced_load(uint64_t iter, const data_t *indices, data_t *indices_fifo);

  void krnl_spmv_reduced_values(uint64_t iter, const data_t *values, data_t *values_fifo);

  void krnl_spmv_reduced_split(data_t *indices_fifo, data_t *rows_fifo, data_t *cols_fifo);

  void krnl_spmv_reduced_MAC(uint64_t iter, const data_t *x, data_t *rows_fifo, data_t *cols_fifo, data_t *values_fifo, data_t *results_fifo);

  void krnl_spmv_reduced_write(uint64_t iter, data_t *y, data_t *results_fifo);

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif