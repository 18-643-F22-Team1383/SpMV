#include "spmv_helper.h"
#include <sys/time.h>

int main(int argc, char *argv[])
{
#ifdef __VITIS_CL__
  if (argc != 4)
  {
    std::cout << "Provide binary in argv[1], batch size in argv[2] & whether to verify in argv[3] (1 or 0)\n";
    return 1;
  }
  std::string binary_file = argv[1];
  uint64_t batch_size = atoi(argv[2]);
  uint64_t enable_verify = atoi(argv[3]);
#else
  uint64_t batch_size = 10;
  uint64_t enable_verify = 1;
#endif
  struct timeval start_time, end_time;
  bool mismatch = false;

  data_t *ptr_values, *ptr_colIdx, *ptr_rowPtr, *ptr_x, *ptr_y;
  data_t *ref_values, *ref_colIdx, *ref_rowPtr, *ref_x, *ref_y;
  // data_t *ptr_indices;
  uintbuswidth_t *ptr_indices, *ptr_kernel_values, *ptr_kernel_y;

  // Compute the size of array in bytes
  // uint64_t num_values = batch_size * NNZ;
  // uint64_t num_colIdx = batch_size * NNZ;
  // uint64_t num_rowPtr = batch_size * (NN + 1);
  // uint64_t num_x = batch_size * MM;
  // uint64_t num_y = batch_size * NN;
  // uint64_t num_indices = batch_size * (NNZ + NN);
  uint64_t num_values = batch_size * NNZ;
  uint64_t num_colIdx = batch_size * NNZ;
  uint64_t num_rowPtr = batch_size * (NN + 1);
  uint64_t num_x = batch_size * MM;
  uint64_t num_y = batch_size * NN;
  uint64_t num_kernel_values = batch_size * (NNZ / MULTI_FACTOR);
  uint64_t num_kernel_y = batch_size * (NN / MULTI_FACTOR);
  uint64_t num_indices = batch_size * ((NNZ + NN) / MULTI_FACTOR);

  cl_object cl_obj;

  krnl_object spmv_obj;
  spmv_obj.index = 0;
  spmv_obj.name = "krnl_spmv_multi";

#ifdef __VITIS_CL__
  std::cout << "===== Initialize device ======" << std::endl;
  initialize_device(cl_obj);

  std::cout << "===== Reading xclbin ======" << std::endl;
  // Read spmv
  read_xclbin(binary_file, cl_obj.bins);

  std::cout << "\n===== Programming kernel ======" << std::endl;
  program_kernel(cl_obj, spmv_obj);
#endif

  std::cout << "\n===== Allocating buffers ======" << std::endl;
  uint64_t buf_idx = 0;
  // Fast
  // allocate_readonly_mem(cl_obj, (void **)&ptr_values, buf_idx++,
  //                       num_values * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_colIdx, buf_idx++,
  //                       num_colIdx * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_rowPtr, buf_idx++,
  //                       num_rowPtr * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_x, buf_idx++,
  //                       num_x * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_y, buf_idx++,
  //                       num_y * sizeof(data_t));

  // Reduced
  // allocate_readonly_mem(cl_obj, (void **)&ptr_values, buf_idx++,
  //                       num_values * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_indices, buf_idx++,
  //                       num_indices * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_x, buf_idx++,
  //                       num_x * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_y, buf_idx++,
  //                       num_y * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_colIdx, buf_idx++,
  //                       num_colIdx * sizeof(data_t));
  // allocate_readonly_mem(cl_obj, (void **)&ptr_rowPtr, buf_idx++,
  //                       num_rowPtr * sizeof(data_t));

  // Multi
  allocate_readonly_mem(cl_obj, (void **)&ptr_kernel_values, buf_idx++,
                        num_kernel_values * sizeof(uintbuswidth_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_indices, buf_idx++,
                        num_indices * sizeof(uintbuswidth_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_x, buf_idx++,
                        num_x * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_kernel_y, buf_idx++,
                        num_kernel_y * sizeof(uintbuswidth_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_colIdx, buf_idx++,
                        num_colIdx * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_rowPtr, buf_idx++,
                        num_rowPtr * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_y, buf_idx++,
                        num_y * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_values, buf_idx++,
                        num_values * sizeof(data_t));

  MALLOC_CHECK(ref_values = new data_t[num_values]);
  MALLOC_CHECK(ref_colIdx = new data_t[num_colIdx]);
  MALLOC_CHECK(ref_rowPtr = new data_t[num_rowPtr]);
  MALLOC_CHECK(ref_x = new data_t[num_x]);
  MALLOC_CHECK(ref_y = new data_t[num_y]);

  // Set randomized inputs in reference copy
  initialize_sparse_matrix(ref_rowPtr, ref_colIdx, batch_size);
  initialize_buffer(ref_values, num_values, true);
  initialize_buffer(ref_x, num_x, true);

  // Reduced
  // uint32_t col_left = 0;
  // uint32_t row_index = 0;
  // uint32_t row_batch = 0;
  // uint32_t col_index = 0;
  // for (uint32_t i = 0; i < num_indices; i++)
  // {
  //   if (col_left == 0)
  //   {
  //     col_left = ref_rowPtr[(NN + 1) * row_batch + row_index + 1] - ref_rowPtr[(NN + 1) * row_batch + row_index];
  //     // printf("Indices %d: row index: %d\n", i, col_left);
  //     ptr_indices[i] = col_left;
  //     row_index++;
  //     if (row_index == NN)
  //     {
  //       row_index = 0;
  //       row_batch++;
  //     }
  //   }
  //   else
  //   {
  //     // printf("Indices %d: col index: %d\n", i, ref_colIdx[col_index]);
  //     ptr_indices[i] = ref_colIdx[col_index];
  //     col_index++;
  //     col_left--;
  //   }
  // }

  // Multi
  uint32_t col_left = 0;
  uint32_t row_index = 0;
  uint32_t row_batch = 0;
  uint32_t col_index = 0;
  for (uint32_t i = 0; i < MULTI_FACTOR; i++)
  {
    for (uint32_t j = 0; j < num_indices; j++)
    {
      if (i == 0)
      {
        if (col_left == 0)
        {
          col_left = ref_rowPtr[(NN + 1) * row_batch + row_index + 1] - ref_rowPtr[(NN + 1) * row_batch + row_index];
          // printf("Indices %d: row index: %d\n", i, col_left);
          ptr_indices[j] = (uintbuswidth_t)col_left;
          row_index++;
          if (row_index == NN)
          {
            row_index = 0;
            row_batch++;
          }
        }
        else
        {
          // printf("Indices %d: col index: %d\n", i, ref_colIdx[col_index]);
          ptr_indices[j] = (uintbuswidth_t)ref_colIdx[col_index];
          col_index++;
          col_left--;
        }
      }
      else
      {
        if (col_left == 0)
        {
          col_left = ref_rowPtr[(NN + 1) * row_batch + row_index + 1] - ref_rowPtr[(NN + 1) * row_batch + row_index];
          // printf("Indices %d: row index: %d\n", i, col_left);
          ptr_indices[j] = ptr_indices[j] | ((uintbuswidth_t)col_left << (DATA_WIDTH * i));
          row_index++;
          if (row_index == NN)
          {
            row_index = 0;
            row_batch++;
          }
        }
        else
        {
          // printf("Indices %d: col index: %d\n", i, ref_colIdx[col_index]);
          ptr_indices[j] = ptr_indices[j] | ((uintbuswidth_t)ref_colIdx[col_index] << (DATA_WIDTH * i));
          col_index++;
          col_left--;
        }
      }
    }
  }

  // for (uint32_t i = 0; i < num_indices; i++)
  // {
  //   uintbuswidth_t idx = ptr_indices[i];
  //   printf("Indices %d: \n\t 0: %d;\n\t 1: %d;\n\t 2: %d;\n\t 3: %d;\n", i,
  //          (data_t)(ptr_indices[i]), (data_t)(ptr_indices[i] >> DATA_WIDTH), (data_t)(ptr_indices[i] >> DATA_WIDTH * 2), (data_t)(ptr_indices[i] >> DATA_WIDTH * 3));
  // }

  // Multi
  uint32_t value_index = 0;
  for (uint32_t i = 0; i < MULTI_FACTOR; i++)
  {
    for (uint32_t j = 0; j < num_kernel_values; j++)
    {
      if (i == 0)
      {
        ptr_kernel_values[j] = (uintbuswidth_t)ref_values[value_index];
        value_index++;
      }
      else
      {
        ptr_kernel_values[j] = ptr_kernel_values[j] | ((uintbuswidth_t)ref_values[value_index] << (DATA_WIDTH * i));
        value_index++;
      }
    }
  }

  // copy ref copy to kernel use copy, converting to kernel expected layout
  for (uint32_t i = 0; i < num_values; i++)
    ptr_values[i] = ref_values[i];
  for (uint32_t i = 0; i < num_colIdx; i++)
    ptr_colIdx[i] = ref_colIdx[i];
  for (uint32_t i = 0; i < num_rowPtr; i++)
    ptr_rowPtr[i] = ref_rowPtr[i];
  for (uint32_t i = 0; i < num_x; i++)
    ptr_x[i] = ref_x[i];

  // Random initialize output for kernel use
  initialize_buffer(ptr_y, num_y, true); // cannot assume 0'ed

  std::cout << "\n===== Execution and Timing starts ======" << std::endl;
  gettimeofday(&start_time, NULL);

#ifdef __VITIS_CL__
  spmv_run_kernel(cl_obj, spmv_obj, batch_size);
#else
  // krnl_spmv(ptr_values, ptr_colIdx, ptr_rowPtr, ptr_x, ptr_y, batch_size);
  // krnl_spmv_fast_V2(ptr_values, ptr_colIdx, ptr_rowPtr, ptr_x, ptr_y, batch_size);
  // krnl_spmv_fast(ptr_values, ptr_colIdx, ptr_rowPtr, ptr_x, ptr_y, batch_size);
  krnl_spmv_reduced(ptr_values, ptr_indices, ptr_x, ptr_y, batch_size);
#endif

  gettimeofday(&end_time, NULL);
  std::cout << "Execution and Timing finished!\n"
            << std::endl;

  // Multi
  uint32_t y_index = 0;
  for (uint32_t i = 0; i < MULTI_FACTOR; i++)
  {
    for (uint32_t j = 0; j < num_kernel_y; j++)
    {
      ptr_y[y_index] = (data_t)(ptr_kernel_y[j] >> (DATA_WIDTH * i));
      y_index++;
    }
  }

  // for (uint32_t i = 0; i < num_kernel_y; i++)
  // {
  //   uintbuswidth_t result = ptr_kernel_y[i];
  //   printf("Result %d: \n\t 0: %d;\n\t 1: %d;\n\t 2: %d;\n\t 3: %d;\n", i,
  //          (data_t)(ptr_kernel_y[i]), (data_t)(ptr_kernel_y[i] >> DATA_WIDTH), (data_t)(ptr_kernel_y[i] >> DATA_WIDTH * 2), (data_t)(ptr_kernel_y[i] >> DATA_WIDTH * 3));
  // }

  if (enable_verify)
  {
    std::cout << "===== Verification starts ======" << std::endl;
    mismatch = spmv_check(ptr_values, ptr_colIdx, ptr_rowPtr, ptr_x, ptr_y,
                          ref_values, ref_colIdx, ref_rowPtr, ref_x, ref_y, batch_size);
    std::cout << "SpMV TEST " << (mismatch ? "FAILED" : "PASSED") << "\n"
              << std::endl;
  }

  delete[] ref_values;
  delete[] ref_colIdx;
  delete[] ref_rowPtr;
  delete[] ref_x;
  delete[] ref_y;

  // Fast
  // deallocate_mem(cl_obj, ptr_values, 0);
  // deallocate_mem(cl_obj, ptr_colIdx, 1);
  // deallocate_mem(cl_obj, ptr_rowPtr, 2);
  // deallocate_mem(cl_obj, ptr_x, 3);
  // deallocate_mem(cl_obj, ptr_y, 4);

  // Reduced
  // deallocate_mem(cl_obj, ptr_values, 0);
  // deallocate_mem(cl_obj, ptr_indices, 1);
  // deallocate_mem(cl_obj, ptr_x, 2);
  // deallocate_mem(cl_obj, ptr_y, 3);
  // deallocate_mem(cl_obj, ptr_colIdx, 4);
  // deallocate_mem(cl_obj, ptr_rowPtr, 5);

  // Reduced
  deallocate_mem(cl_obj, ptr_kernel_values, 0);
  deallocate_mem(cl_obj, ptr_indices, 1);
  deallocate_mem(cl_obj, ptr_x, 2);
  deallocate_mem(cl_obj, ptr_kernel_y, 3);
  deallocate_mem(cl_obj, ptr_colIdx, 4);
  deallocate_mem(cl_obj, ptr_rowPtr, 5);
  deallocate_mem(cl_obj, ptr_y, 6);
  deallocate_mem(cl_obj, ptr_values, 7);

  std::cout << "===== Reporting measured throughput ======" << std::endl;
  float timeusec = (end_time.tv_sec - start_time.tv_sec) * 1e6 + (end_time.tv_usec - start_time.tv_usec);
  printf("Runtime = %0.1f (microsec) \n\n", timeusec);
  double num_operations = batch_size * (double)2.0 * NNZ;
  printf("# of operations = %.0f\n", num_operations);
  printf("Throughput: %.5f GigaOP/sec\n",
         (double)1.0e-3 * num_operations / timeusec);

  std::cout << "\n===== Exiting ======" << std::endl;
  return mismatch;
}
