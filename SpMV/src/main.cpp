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

  // Compute the size of array in bytes
  uint64_t num_values = batch_size * NNZ;
  uint64_t num_colIdx = batch_size * NNZ;
  uint64_t num_rowPtr = batch_size * (NN + 1);
  uint64_t num_x = batch_size * MM;
  uint64_t num_y = batch_size * NN;

  cl_object cl_obj;

  krnl_object spmv_obj;
  spmv_obj.index = 0;
  spmv_obj.name = "krnl_spmv";

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
  allocate_readonly_mem(cl_obj, (void **)&ptr_values, buf_idx++,
                        num_values * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_colIdx, buf_idx++,
                        num_colIdx * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_rowPtr, buf_idx++,
                        num_rowPtr * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_x, buf_idx++,
                        num_x * sizeof(data_t));
  allocate_readonly_mem(cl_obj, (void **)&ptr_y, buf_idx++,
                        num_y * sizeof(data_t));

  MALLOC_CHECK(ref_values = new data_t[num_values]);
  MALLOC_CHECK(ref_colIdx = new data_t[num_colIdx]);
  MALLOC_CHECK(ref_rowPtr = new data_t[num_rowPtr]);
  MALLOC_CHECK(ref_x = new data_t[num_x]);
  MALLOC_CHECK(ref_y = new data_t[num_y]);

  // Set randomized inputs in reference copy
  initialize_sparse_matrix(ref_rowPtr, ref_colIdx, batch_size);
  initialize_buffer(ref_values, num_values, true);
  initialize_buffer(ref_x, num_x, true);

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
  //krnl_spmv(ptr_values, ptr_colIdx, ptr_rowPtr, ptr_x, ptr_y, batch_size);
  //krnl_spmv_fast_V2(ptr_values, ptr_colIdx, ptr_rowPtr, ptr_x, ptr_y, batch_size);
  krnl_spmv_fast(ptr_values, ptr_colIdx, ptr_rowPtr, ptr_x, ptr_y, batch_size);
#endif

  gettimeofday(&end_time, NULL);
  std::cout << "Execution and Timing finished!\n"
            << std::endl;

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

  deallocate_mem(cl_obj, ptr_values, 0);
  deallocate_mem(cl_obj, ptr_colIdx, 1);
  deallocate_mem(cl_obj, ptr_rowPtr, 2);
  deallocate_mem(cl_obj, ptr_x, 3);
  deallocate_mem(cl_obj, ptr_y, 4);

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
