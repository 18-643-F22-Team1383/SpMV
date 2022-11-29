#include "krnl_spmv.h"
#include <ap_int.h>
#include <hls_stream.h>

// #ifdef __VITIS_CL__
// extern "C"
// {
// #endif
void krnl_spmv(const data_t *values, const data_t *colIdx, const data_t *rowPtr,
               const data_t *x, data_t *y, uint64_t batch_size)
{

  for (index_t iter = 0; iter < batch_size; iter++)
  {
    for (index_t i = 0; i < NN; i++)
    {
      data_t yt = 0;
      for (index_t k = ARRAY2(rowPtr, iter, i, NN + 1); k < ARRAY2(rowPtr, iter, i + 1, NN + 1); k++)
      {
        yt += ARRAY2(values, iter, k, NNZ) *
              ARRAY2(x, iter, (ARRAY2(colIdx, iter, k, NNZ)), MM);
      }
      ARRAY2(y, iter, i, NN) = yt;
    }
  }
}
// #ifdef __VITIS_CL__ // for lab 3
// } // extern
// #endif

/**
 * @default: sparse matrix has (n) rows and (m) columns totoally (NNZ) non-zero elements
 * @param:  valeus: all the non-zero values extracted from input sparse matrix
 *          col_indx: all the non-zero values corresponding column index, len(col_index) = NNZ
 *          rowPtr: row pointer stores accumulative non-zero values before current row, len(rowPtr) = n + 1
 *          row_length: num of non-zero elements of each row. row_length[i] = rowPtr[i+1] - rowPtr[i]. len(row_length) = n
 *          x: dense vector, len(x) = n
 *          y: output vector, len(y) = n
 *          NNZ: non-zero elements of input sparse matrix, also is the size of input values[]
 *          NN: num of rows of input sparse matrix. NN = n
 * @summary: reduced port stream SpMV kernel
 */
#ifdef __VITIS_CL__
extern "C"
{
#endif
  void krnl_spmv_reduced(const data_t *values, const data_t *indices,
                         const data_t *x, data_t *y, uint64_t batch_size)
  {

// initialize the fifos and data stream
#pragma HLS DATAFLOW
    const data_t indices_fifo_depth = NNZ + NN;
    const data_t rows_fifo_depth = NN;
    const data_t values_fifo_depth = NNZ;
    const data_t cols_fifo_depth = NNZ;
    const data_t results_fifo_depth = NN;
    // defines all the fifos
    hls::stream<data_t> indices_fifo;
#pragma HLS STREAM variable = indices_fifo depth = indices_fifo_depth type = fifo

    hls::stream<data_t> rows_fifo;
#pragma HLS STREAM variable = rows_fifo depth = rows_fifo_depth type = fifo

    hls::stream<data_t> values_fifo;
#pragma HLS STREAM variable = values_fifo depth = values_fifo_depth type = fifo

    hls::stream<data_t> cols_fifo;
#pragma HLS STREAM variable = cols_fifo depth = cols_fifo_depth type = fifo

    hls::stream<data_t> results_fifo;
#pragma HLS STREAM variable = results_fifo depth = results_fifo_depth type = fifo

    // batch_size iteration
    for (uint64_t iter = 0; iter < batch_size; iter++)
    {

      // feed data into indices
      for (index_t i = 0; i < NNZ + NN; i++)
      {
#pragma HLS pipeline
        indices_fifo << ARRAY2(indices, iter, i, NNZ + NN);
      }
      data_t idx_col_left = 0;
      // feed data into the fifos
      // feed row index into fifo
      for (index_t i = 0; i < NNZ + NN; i++)
      {
#pragma HLS PIPELINE
        data_t indices = indices_fifo.read();
        if (idx_col_left == 0)
        {
          idx_col_left = indices;
          rows_fifo << idx_col_left;
        }
        else
        {
          cols_fifo << indices;
          idx_col_left--;
        }
      }

      // feed column index and values into fifo
      for (index_t i = 0; i < NNZ; i++)
      {
#pragma HLS pipeline
        values_fifo << ARRAY2(values, iter, i, NNZ);
      }

      /**
       *  initialize the read parameters
       */
      data_t col_left;
      data_t row_left;

      data_t accumulator;
      data_t value;
      data_t col;

      // read from fifo -> apply multiplication -> store value in local buffer -> write back
//       for (index_t i = 0; i < NN; i++)
//       {
// #pragma HLS pipeline
//         col_left = rows_fifo.read();
//         accumulator = 0;
//         for (index_t j = 0; j < col_left; j++)
//         {
// #pragma HLS pipeline
//           // multiply and accumulate
//           value = values_fifo.read();
//           col = cols_fifo.read();
//           accumulator += value * ARRAY2(x, iter, col, MM);
//         }
//         results_fifo << accumulator;
//       }
      for (index_t i = 0; i < NNZ+NN; i++)
      {
#pragma HLS pipeline
        // read parameters from fifos
        if (i == 0 || col_left == 0)
        {
          col_left = rows_fifo.read();
          accumulator = 0;
        } else {
          // multiply and accumulate
          value = values_fifo.read();
          col = cols_fifo.read();
          accumulator += value * ARRAY2(x, iter, col, MM);

          col_left--;

          // write back the dot product to fifo
          if (col_left == 0)
          {
            results_fifo << accumulator;
          }
        }
      }
      // write back the accumulation to y vector
      for (index_t i = 0; i < NN; i++)
      {
#pragma HLS pipeline
        ARRAY2(y, iter, i, NN) = results_fifo.read();
      }
    }
  }
#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif

/**
 * @default: sparse matrix has (n) rows and (m) columns totoally (NNZ) non-zero elements
 * @param:  valeus: all the non-zero values extracted from input sparse matrix
 *          col_indx: all the non-zero values corresponding column index, len(col_index) = NNZ
 *          rowPtr: row pointer stores accumulative non-zero values before current row, len(rowPtr) = n + 1
 *          row_length: num of non-zero elements of each row. row_length[i] = rowPtr[i+1] - rowPtr[i]. len(row_length) = n
 *          x: dense vector, len(x) = n
 *          y: output vector, len(y) = n
 *          NNZ: non-zero elements of input sparse matrix, also is the size of input values[]
 *          NN: num of rows of input sparse matrix. NN = n
 * @summary: fast stream SpMV kernel
 */
#ifdef __VITIS_CL__
extern "C"
{
#endif
  void krnl_spmv_fast(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                      const data_t *x, data_t *y, uint64_t batch_size = 10)
  {

// initialize the fifos and data stream
#pragma HLS DATAFLOW
    const data_t rows_fifo_depth = NN;
    const data_t values_fifo_depth = NNZ;
    const data_t cols_fifo_depth = NNZ;
    const data_t results_fifo_depth = NN;
    // defines all the fifos
    hls::stream<data_t> rows_fifo;
#pragma HLS STREAM variable = rows_fifo depth = rows_fifo_depth type = fifo

    hls::stream<data_t> values_fifo;
#pragma HLS STREAM variable = values_fifo depth = values_fifo_depth type = fifo

    hls::stream<data_t> cols_fifo;
#pragma HLS STREAM variable = cols_fifo depth = cols_fifo_depth type = fifo

    hls::stream<data_t> results_fifo;
#pragma HLS STREAM variable = results_fifo depth = results_fifo_depth type = fifo

    // batch_size iteration
    for (uint64_t iter = 0; iter < batch_size; iter++)
    {

      // initialize row_length vector
      data_t row_length[NN];
      for (index_t i = 0; i < NN; i++)
      {
#pragma HLS pipeline
        row_length[i] = ARRAY2(rowPtr, iter, i + 1, NN + 1) - ARRAY2(rowPtr, iter, i, NN + 1);
      }

      // feed data into the fifos
      // feed row index into fifo
      for (index_t i = 0; i < NN; i++)
      {
#pragma HLS pipeline
        rows_fifo << row_length[i];
      }

      // feed column index and values into fifo
      for (index_t i = 0; i < NNZ; i++)
      {
#pragma HLS pipeline
        values_fifo << ARRAY2(values, iter, i, NNZ);
        cols_fifo << ARRAY2(col_index, iter, i, NNZ);
      }

      /**
       *  initialize the read parameters
       */
      data_t col_left;
      data_t row_left;

      data_t accumulator;
      data_t value;
      data_t col;

      // read from fifo -> apply multiplication -> store value in local buffer -> write back
      // for (index_t i = 0; i < NN; i++)
      // {
      //   col_left = rows_fifo.read();
      //   accumulator = 0;
      //   for (index_t j = 0; j < col_left; j++)
      //   {
      //     // multiply and accumulate
      //     value = values_fifo.read();
      //     col = cols_fifo.read();
      //     accumulator += value * ARRAY2(x, iter, col, MM);
      //   }
      //   results_fifo << accumulator;
      // }
      for (index_t i = 0; i < NNZ; i++)
      {
#pragma HLS pipeline
        // read parameters from fifos
        if (i == 0 || col_left == 0)
        {
          col_left = rows_fifo.read();
          accumulator = 0;
        }

        // multiply and accumulate
        value = values_fifo.read();
        col = cols_fifo.read();
        accumulator += value * ARRAY2(x, iter, col, MM);

        col_left--;

        // write back the dot product to fifo
        if (col_left == 0)
        {
          results_fifo << accumulator;
        }
      }

      // write back the accumulation to y vector
      for (index_t i = 0; i < NN; i++)
      {
#pragma HLS pipeline
        ARRAY2(y, iter, i, NN) = results_fifo.read();
      }
    }
  }

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif
