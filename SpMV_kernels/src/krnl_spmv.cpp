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
    // defines all the fifos
    // hls::stream<data_t, indices_fifo_depth> indices_fifo;
    // #pragma HLS STREAM variable = indices_fifo depth = indices_fifo_depth type = fifo

    // hls::stream<data_t, rows_fifo_depth> rows_fifo;
    // #pragma HLS STREAM variable = rows_fifo depth = rows_fifo_depth type = fifo

    // hls::stream<data_t, values_fifo_depth> values_fifo;
    // #pragma HLS STREAM variable = values_fifo depth = values_fifo_depth type = fifo

    // hls::stream<data_t, cols_fifo_depth> cols_fifo;
    // #pragma HLS STREAM variable = cols_fifo depth = cols_fifo_depth type = fifo

    // hls::stream<data_t, results_fifo_depth> results_fifo;
    // #pragma HLS STREAM variable = results_fifo depth = results_fifo_depth type = fifo

// batch_size iteration
#pragma HLS interface mode = m_axi port = values bundle = gmem
#pragma HLS interface mode = m_axi port = indices bundle = gmem0
#pragma HLS interface mode = m_axi port = x bundle = gmem1
#pragma HLS interface mode = m_axi port = y bundle = gmem

    for (uint64_t iter = 0; iter < batch_size; iter++)
    {
      //#pragma HLS pipeline off

#pragma HLS DATAFLOW

      const data_t indices_fifo_depth = 2;
      const data_t rows_fifo_depth = 2;
      const data_t values_fifo_depth = 2;
      const data_t cols_fifo_depth = 2;
      const data_t results_fifo_depth = 2;
      data_t indices_fifo[NNZ + NN], rows_fifo[NN], values_fifo[NNZ], cols_fifo[NNZ], results_fifo[NN];

#pragma HLS STREAM variable = indices_fifo depth = indices_fifo_depth type = fifo
#pragma HLS STREAM variable = rows_fifo depth = rows_fifo_depth type = fifo
#pragma HLS STREAM variable = values_fifo depth = values_fifo_depth type = fifo
#pragma HLS STREAM variable = cols_fifo depth = cols_fifo_depth type = fifo
#pragma HLS STREAM variable = results_fifo depth = results_fifo_depth type = fifo

      krnl_spmv_reduced_load(iter, indices, indices_fifo);
      krnl_spmv_reduced_values(iter, values, values_fifo);
      krnl_spmv_reduced_split(indices_fifo, rows_fifo, cols_fifo);
      krnl_spmv_reduced_MAC(iter, x, rows_fifo, cols_fifo, values_fifo, results_fifo);
      krnl_spmv_reduced_write(iter, y, results_fifo);
    }
  }

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
  void krnl_spmv_fast(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                      const data_t *x, data_t *y, uint64_t batch_size = 10)
  {

    // // initialize the fifos and data stream
    // #pragma HLS DATAFLOW
    //     const data_t rows_fifo_depth = NN;
    //     const data_t values_fifo_depth = NNZ;
    //     const data_t cols_fifo_depth = NNZ;
    //     const data_t results_fifo_depth = NN;
    //     // defines all the fifos
    //     hls::stream<data_t> rows_fifo;
    // #pragma HLS STREAM variable = rows_fifo depth = rows_fifo_depth type = fifo

    //     hls::stream<data_t> values_fifo;
    // #pragma HLS STREAM variable = values_fifo depth = values_fifo_depth type = fifo

    //     hls::stream<data_t> cols_fifo;
    // #pragma HLS STREAM variable = cols_fifo depth = cols_fifo_depth type = fifo

    //     hls::stream<data_t> results_fifo;
    // #pragma HLS STREAM variable = results_fifo depth = results_fifo_depth type = fifo

    //     // batch_size iteration
    //     for (uint64_t iter = 0; iter < batch_size; iter++)
    //     {

    //       // initialize row_length vector
    //       data_t row_length[NN];
    //       for (index_t i = 0; i < NN; i++)
    //       {
    // #pragma HLS pipeline
    //         row_length[i] = ARRAY2(rowPtr, iter, i + 1, NN + 1) - ARRAY2(rowPtr, iter, i, NN + 1);
    //       }

    //       // feed data into the fifos
    //       // feed row index into fifo
    //       for (index_t i = 0; i < NN; i++)
    //       {
    // #pragma HLS pipeline
    //         rows_fifo << row_length[i];
    //       }

    //       // feed column index and values into fifo
    //       for (index_t i = 0; i < NNZ; i++)
    //       {
    // #pragma HLS pipeline
    //         values_fifo << ARRAY2(values, iter, i, NNZ);
    //         cols_fifo << ARRAY2(col_index, iter, i, NNZ);
    //       }

    //       /**
    //        *  initialize the read parameters
    //        */
    //       data_t col_left;
    //       data_t row_left;

    //       data_t accumulator;
    //       data_t value;
    //       data_t col;

    //       // read from fifo -> apply multiplication -> store value in local buffer -> write back
    //       // for (index_t i = 0; i < NN; i++)
    //       // {
    //       //   col_left = rows_fifo.read();
    //       //   accumulator = 0;
    //       //   for (index_t j = 0; j < col_left; j++)
    //       //   {
    //       //     // multiply and accumulate
    //       //     value = values_fifo.read();
    //       //     col = cols_fifo.read();
    //       //     accumulator += value * ARRAY2(x, iter, col, MM);
    //       //   }
    //       //   results_fifo << accumulator;
    //       // }
    //       for (index_t i = 0; i < NNZ; i++)
    //       {
    // #pragma HLS pipeline
    //         // read parameters from fifos
    //         if (i == 0 || col_left == 0)
    //         {
    //           col_left = rows_fifo.read();
    //           accumulator = 0;
    //         }

    //         // multiply and accumulate
    //         value = values_fifo.read();
    //         col = cols_fifo.read();
    //         accumulator += value * ARRAY2(x, iter, col, MM);

    //         col_left--;

    //         // write back the dot product to fifo
    //         if (col_left == 0)
    //         {
    //           results_fifo << accumulator;
    //         }
    //       }

    //       // write back the accumulation to y vector
    //       for (index_t i = 0; i < NN; i++)
    //       {
    // #pragma HLS pipeline
    //         ARRAY2(y, iter, i, NN) = results_fifo.read();
    //       }
    //     }
  }

  void krnl_spmv_reduced_load(uint64_t iter, const data_t *indices, data_t *indices_fifo)
  {
#pragma HLS inline off
    for (index_t i = 0; i < NNZ + NN; i++)
    {
#pragma HLS pipeline
      indices_fifo[i] = ARRAY2(indices, iter, i, NNZ + NN);
    }
  }

  void krnl_spmv_reduced_values(uint64_t iter, const data_t *values, data_t *values_fifo)
  {
#pragma HLS inline off
    // feed column index and values into fifo
    for (index_t i = 0; i < NNZ; i++)
    {
#pragma HLS pipeline
      values_fifo[i] = ARRAY2(values, iter, i, NNZ);
    }
  }

  void krnl_spmv_reduced_split(data_t *indices_fifo, data_t *rows_fifo, data_t *cols_fifo)
  {
#pragma HLS inline off
    data_t col_left = 0;
    data_t rows_idx = 0;
    data_t cols_idx = 0;
    // feed data into the fifos
    // feed row index into fifo
    for (index_t i = 0; i < NNZ + NN; i++)
    {
#pragma HLS PIPELINE
      data_t index = indices_fifo[i];
      if (col_left == 0)
      {
        col_left = index;
        rows_fifo[rows_idx] = col_left;
        rows_idx++;
      }
      else
      {
        cols_fifo[cols_idx] = index;
        col_left--;
        cols_idx++;
      }
    }
  }

  void krnl_spmv_reduced_MAC(uint64_t iter, const data_t *x, data_t *rows_fifo, data_t *cols_fifo, data_t *values_fifo, data_t *results_fifo)
  {
#pragma HLS inline off
    /**
     *  initialize the read parameters
     */
    data_t col_left = 0;
    data_t rows_idx = 0;
    data_t cols_idx = 0;
    data_t values_idx = 0;
    data_t accumulator;
    data_t value;
    data_t col;

    for (index_t i = 0; i < NNZ + NN; i++)
    {
#pragma HLS pipeline
      // read parameters from fifos
      if (i == 0 || col_left == 0)
      {
        col_left = rows_fifo[rows_idx];
        accumulator = 0;
        rows_idx++;
      }
      else
      {
        // multiply and accumulate
        value = values_fifo[cols_idx];
        col = cols_fifo[cols_idx];
        cols_idx++;
        accumulator += value * ARRAY2(x, iter, col, MM);

        col_left--;
        // write back the dot product to fifo
      }
      if (col_left == 0)
      {
        results_fifo[values_idx] = accumulator;
        values_idx++;
      }
    }
  }

  void krnl_spmv_reduced_write(uint64_t iter, data_t *y, data_t *results_fifo)
  {
#pragma HLS inline off
    // write back the accumulation to y vector
    for (index_t i = 0; i < NN; i++)
    {
#pragma HLS pipeline
      ARRAY2(y, iter, i, NN) = results_fifo[i];
    }
  }

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif
