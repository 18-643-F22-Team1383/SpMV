#pragma once

#include "util643.h"
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#ifdef __VITIS_CL__
#include <CL/cl2.hpp>
#endif
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

#define OCL_CHECK(error, call)                                                                      \
    call;                                                                                           \
    if (error != CL_SUCCESS) {                                                                      \
        printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error);    \
        exit(EXIT_FAILURE);                                                                         \
    }

#define MALLOC_CHECK(call)                                                  \
    if ((call) == NULL) {                                                   \
        printf("%s:%d Error calling " #call "\n", __FILE__, __LINE__);      \
        exit(EXIT_FAILURE);                                                 \
    }

#ifndef __VITIS_CL__
typedef int cl_int;
#endif

// Common OpenCL objects
typedef struct cl_object {
#ifdef __VITIS_CL__
    cl::Device device;
    cl::Context context;
    cl::CommandQueue q;
    cl::Program::Binaries bins;

    // Program/Kernel object needs to be destructed
    // before instantiating a new object
    cl::Program *program = NULL;
    cl::Kernel *krnl = NULL;
    std::vector<cl::Buffer> buffers;
#endif
} cl_object;

// Kernel-specific OpenCL objects
typedef struct krnl_object {
    std::string name;
    unsigned index;
} krnl_object;

#ifdef __VITIS_CL__
// Find and initialize the device, context and command queues
void initialize_device(cl_object &obj);

// Read xclbin files into memory
void read_xclbin(std::string xclbinFilename, cl::Program::Binaries &bins);

// DFX Kernel into the dynamic region and obtain the kernel object
void program_kernel(cl_object &cl_obj, krnl_object &krnl_obj);
#endif

// Allocate read-only memory on device and map pointers into the host
void allocate_readonly_mem (cl_object &cl_obj, void **ptr, uint64_t idx,
                            uint64_t size_in_bytes);

// Allocate read-write memory on device and map pointers into the host
void allocate_readwrite_mem (cl_object &cl_obj, void **ptr, uint64_t idx,
                             uint64_t size_in_bytes);

// Unmap device memory when done
void deallocate_mem (cl_object &cl_obj, void *ptr, uint64_t idx);
