#include "spmv_helper.h"
#include <string.h>

// Set kernel arguments and execute it
#ifdef __VITIS_CL__
void spmv_run_kernel(cl_object &cl_obj, krnl_object &krnl_obj, uint64_t batch_size) {
    cl_int err;

    std::cout << "Running kernel for spmv..." << std::endl;

    // Get i/o buffers from kernel object
    cl::Buffer *buffer_values = &cl_obj.buffers[0];
    cl::Buffer *buffer_colIdx = &cl_obj.buffers[1];
    cl::Buffer *buffer_rowPtr = &cl_obj.buffers[2];
    cl::Buffer *buffer_x = &cl_obj.buffers[3];
    cl::Buffer *buffer_y = &cl_obj.buffers[4];

    // Set the kernel Arguments
    uint64_t narg = 0;
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_values));        // values
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_colIdx));        // columnIndex
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_rowPtr));        // rowPtr
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_x));             // vector X
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, *buffer_y));             // vector Y
    OCL_CHECK(err, err = cl_obj.krnl->setArg(narg++, (uint64_t) batch_size)); // batch size

    // Data will be migrated to kernel space
    OCL_CHECK(err, err = cl_obj.q.enqueueMigrateMemObjects({
            *buffer_values, *buffer_colIdx, *buffer_rowPtr, *buffer_x
        }, 0/* 0 means from host*/));

    // Launch the Kernel; this is nonblocking.
    OCL_CHECK(err, err = cl_obj.q.enqueueTask(*cl_obj.krnl));

    // The result of the previous kernel execution will need to be retrieved in
    // order to view the results. This call will transfer the data from FPGA to
    // source_results vector
    OCL_CHECK(err, cl_obj.q.enqueueMigrateMemObjects({*buffer_y}, CL_MIGRATE_MEM_OBJECT_HOST));

    // Wait for all tasks to finish
    OCL_CHECK(err, cl_obj.q.finish());

    std::cout << "Kernel executions completed" << std::endl;
}
#endif

// Verify a single batch of data
bool verify(data_t *ref, data_t *checkit, uint64_t iter) {
    for (uint64_t i = 0; i < NN; i++) {
        data_t refval = ARRAY2(ref, iter, i, NN);
        data_t checkval = ARRAY2(checkit, iter, i, NN);
        if (!nearlyEqual(checkval, refval)) {
            printf("\n***Result does not match reference: "
                    "iter = %lu, refval = %u, checkval = %u***\n", iter, refval, checkval);
            return 0;
        }
    }
    return 1;
}

bool spmv_check(data_t *ptr_values, data_t *ptr_colIdx, data_t *ptr_rowPtr,
        data_t *ptr_x, data_t *ptr_y,
        data_t *ref_values, data_t *ref_colIdx, data_t *ref_rowPtr,
        data_t *ref_x, data_t *ref_y,
        uint64_t batch_size) {
    std::cout << "Verifying SpMV result..." << std::endl;

    // Verify the result
    uint64_t mismatch = 0;
    uint64_t iter;

    for(iter = 0; iter < batch_size; iter++) {
        spmv_ref_code(ref_values, ref_colIdx, ref_rowPtr, ref_x, ref_y, iter);
        if (!verify(ref_y, ptr_y, iter)) {
            mismatch = 1;
            break;
        }
    }
    return mismatch;
}

void initialize_sparse_matrix(data_t *rowPtr, data_t *colIdx, uint64_t batch_size) {
    for (uint64_t i = 0; i < batch_size; i++) {
        // todo Use a more general random sparse matrix generating algorithm
        for (int i = 0; i < NN + 1; i++) {
            rowPtr[i] = i;
        }
        for (int i = 0; i < NNZ; i++) {
            colIdx[i] = NNZ - i - 1;
        }
    }
}

void initialize_buffer(data_t *ptr, unsigned size, bool notzero) {
    for (unsigned i = 0; i < size; i++) {
        ptr[i] = notzero ? (rand() % VRANGE) : 0;
    }
}

// Reference SpMV code
void spmv_ref_code(data_t *values, data_t *colIdx, data_t *rowPtr,
        data_t *x, data_t *y, uint64_t iter) {
    for (int i = 0; i < NN; i++) {
        data_t yt = 0;
        for (int k = ARRAY2(rowPtr,iter,i,NN+1); k < ARRAY2(rowPtr,iter,i+1,NN+1); k++) {
            yt += ARRAY2(values,iter,k,NNZ) *
                ARRAY2(x,iter,(ARRAY2(colIdx,iter,k,NNZ)),MM);
        }
        ARRAY2(y,iter,i,NN) = yt;
    }
}
