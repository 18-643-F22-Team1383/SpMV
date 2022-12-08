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

  void krnl_spmv_fast(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                      const data_t *x, data_t *y, uint64_t batch_size);

  void krnl_spmv_reduced(const data_t *values, const data_t *indices,
                         const data_t *x, data_t *y, uint64_t batch_size);

  void krnl_spmv_multi(const uintbuswidth_t *values, const uintbuswidth_t *indices,
                       const data_t *x, uintbuswidth_t *y, uint64_t batch_size);

  void krnl_spmv_reduced_load(uint64_t iter, const data_t *indices, data_t *indices_fifo);
  void krnl_spmv_reduced_values(uint64_t iter, const data_t *values, data_t *values_fifo);
  void krnl_spmv_reduced_split(data_t *indices_fifo, data_t *rows_fifo, data_t *cols_fifo);
  void krnl_spmv_reduced_MAC(uint64_t iter, const data_t *x, data_t *rows_fifo, data_t *cols_fifo, data_t *values_fifo, data_t *results_fifo);
  void krnl_spmv_reduced_write(uint64_t iter, data_t *y, data_t *results_fifo);

  void krnl_spmv_multi_dataflow(uint64_t iter, const uintbuswidth_t *values, const uintbuswidth_t *indices,
                                data_t *x_tmp_0, data_t *x_tmp_1, data_t *x_tmp_2, data_t *x_tmp_3, uintbuswidth_t *y);
  void krnl_spmv_multi_load(uint64_t iter, const uintbuswidth_t *indices, data_t *indices_fifo_0, data_t *indices_fifo_1, data_t *indices_fifo_2, data_t *indices_fifo_3);
  void krnl_spmv_multi_values(uint64_t iter, const uintbuswidth_t *values, data_t *values_fifo_0, data_t *values_fifo_1, data_t *values_fifo_2, data_t *values_fifo_3);
  void krnl_spmv_multi_split(data_t *indices_fifo, data_t *rows_fifo, data_t *cols_fifo);
  void krnl_spmv_multi_MAC(uint64_t iter, data_t *x_tmp, data_t *rows_fifo, data_t *cols_fifo, data_t *values_fifo, data_t *results_fifo);
  void krnl_spmv_multi_write(uint64_t iter, uintbuswidth_t *y, data_t *results_fifo_0, data_t *results_fifo_1, data_t *results_fifo_2, data_t *results_fifo_3);

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif
