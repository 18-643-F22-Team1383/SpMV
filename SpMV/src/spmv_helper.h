#pragma once

#include "../../SpMV_kernels/src/krnl_spmv.h"
#include "utils.h"
#include "instance643.h"
#include <sys/time.h>

#ifdef __VITIS_CL__
// Set kernel arguments and execute it
void spmv_run_kernel(cl_object &cl_obj, krnl_object &krnl_obj, uint64_t batch_size);
#endif

// Verification functions
bool spmv_check(data_t *ptr_values, data_t *ptr_colIdx, data_t *ptr_rowPtr,
        data_t *ptr_x, data_t *ptr_y,
        data_t *ref_values, data_t *ref_colIdx, data_t *ref_rowPtr,
        data_t *ref_x, data_t *ref_y,
        uint64_t batch_size);

// Initialize random sparse matrix
void initialize_sparse_matrix(data_t *rowPtr, data_t *colIdx, uint64_t batch_size);

// Initialize memory with random numbers
void initialize_buffer(data_t *ptr, unsigned size, bool notzero);

// Reference SpMV code
void spmv_ref_code(data_t *values, data_t *colIdx, data_t *rowPtr,
        data_t *x, data_t *y, uint64_t iter);
