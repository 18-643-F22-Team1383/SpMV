#include "krnl_spmv.h"
#include <ap_int.h>
#include <hls_stream.h>

#ifdef __VITIS_CL__
extern "C"
{
#endif
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
void krnl_spmv_fast(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                    const data_t *x, data_t *y, uint64_t batch_size)
{

// initialize the fifos and data stream
#pragma HLS DATAFLOW
  const data_t fifo_depth = NNZ;
  // defines all the fifos
  hls::stream<data_t> rows_fifo;
#pragma HLS STREAM variable = rows_fifo depth = fifo_depth type = fifo

  hls::stream<data_t> values_fifo;
#pragma HLS STREAM variable = values_fifo depth = fifo_depth type = fifo

  hls::stream<data_t> cols_fifo;
#pragma HLS STREAM variable = cols_fifo depth = fifo_depth type = fifo

  hls::stream<data_t> results_fifo;
#pragma HLS STREAM variable = results_fifo depth = fifo_depth type = fifo

  hls::stream<data_t> data_length_fifo;
#pragma HLS STREAM variable = data_length_fifo depth = fifo_depth type = fifo

  hls::stream<data_t> row_index_fifo;
#pragma HLS STREAM variable = row_index_fifo depth = fifo_depth type = fifo

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
      data_t row_size = row_length[i];
      rows_fifo << row_size;
    }

    // feed column index and values into fifo
    for (index_t i = 0; i < NNZ; i++)
    {
#pragma HLS pipeline
      values_fifo << ARRAY2(values, iter, 1, NNZ);
      data_t col = ARRAY2(col_index, iter, i, NNZ);
      cols_fifo << col;
    }

    /**
     *  initialize the read parameters
     */
    data_t col_left;
    data_t row_left;

    data_t accumulator;
    data_t value;
    data_t col;

    const data_t II = 1;
    data_t tmp[II];

    // read from fifo -> apply multiplication -> store value in local buffer -> write back
    for (index_t i = 0; i < NNZ; i += II)
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

      col_left -= II;
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
void krnl_spmv_reduced(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                       const data_t *x, data_t *y, uint64_t batch_size)
{

  for (uint64_t iter = 0; iter < batch_size; iter++)
  {
    // initialize row_length vector
    data_t row_length[NN];
    for (index_t i = 0; i < NN; i++)
    {
#pragma HLS pipeline
      row_length[i] = ARRAY2(rowPtr, iter, i + 1, NN + 1) - ARRAY2(rowPtr, iter, i, NN + 1);
    }

    data_t indiciePtr = 0;
    /**
     * Initialize the indices fifo and value fifo
     */
    const data_t fifo_depth = NNZ;

    hls::stream<data_t> indices_fifo;
#pragma HLS STREAM variable = indices_fifo depth = fifo_depth type = fifo

    hls::stream<data_t> values_fifo;
#pragma HLS STREAM variable = values_fifo depth = fifo_depth type = fifo

    hls::stream<data_t> row_fifo;
#pragma HLS STREAM variable = row_fifo depth = fifo_depth type = fifo

    hls::stream<data_t> col_fifo;
#pragma HLS STREAM variable = col_fifo depth = fifo_depth type = fifo

    hls::stream<data_t> values_indices_fifo;
#pragma HLS STREAM variable = values_indices_fifo depth = fifo_depth type = fifo

    hls::stream<data_t> results_fifo;
#pragma HLS STREAM variable = results_fifo depth = fifo_depth type = fifo

    data_t sum;
    const data_t II = 1;
    data_t value;
    data_t term[II];
    data_t col;

    // initialize the indice array
    data_t compact_size = NN + NNZ;
    data_t indices[compact_size];
    for (index_t i = 0; i < NN; i++)
    {
      // fill the row_length element into indeices
      data_t numElment = row_length[i];
      data_t colLowerBound = rowPtr[i];
      data_t colUpperBound = rowPtr[i + 1];
      indices[indiciePtr] = numElment;
      // move indices ptr 1 unit right
      indiciePtr++;
      for (index_t k = colLowerBound; k < colUpperBound; k++, indiciePtr++)
      {
        indices[indiciePtr] = ARRAY2(col_index, iter, k, NNZ);
      }
    }

    // feed the indices
    for (index_t i = 0; i < compact_size; i++)
    {
#pragma HLS pipeline
      indices_fifo << indices[i];
    }
    // feed values
    for (index_t i = 0; i < NNZ; i++)
    {
#pragma HLS pipeline
      values_fifo << ARRAY2(values, iter, i, NNZ);
    }

    data_t col_left = 0;
    // feed indices
    for (index_t i = 0; i < compact_size; i++)
    {
#pragma HLS pipeline
      data_t indices = indices_fifo.read();
      if (col_left == 0)
      {
        // initially the first element in indices fifo
        col_left = indices;
        row_fifo << col_left;
      }
      else
      {
        col_fifo << indices;
        col_left--;
      }
    }

    // reset the col_left
    col_left = 0;
    for (index_t k = 0; k < NNZ + NN * II; k += II)
    {
#pragma HLS pipeline
      if (col_left == 0)
      {
        col_left = row_fifo.read();
        sum = 0;
      }
      else
      {
        for (index_t i = 0; i < II; i++)
        {

          // multiply
          value = values_fifo.read();
          col = col_fifo.read();
          term[i] = value * ARRAY2(x, iter, col, MM);
        }

        data_t sum_tmp = 0;
        for (index_t i = 0; i < II; i += II)
        {
          sum_tmp += term[i];
        }
        sum += sum_tmp;
        col_left -= II;

        if (col_left == 0)
        {
          results_fifo << sum;
        }
      }
    }

    // write back  accumulator to y vector
    for (index_t i = 0; i < NN; i++)
    {
#pragma HLS pipeline
      ARRAY2(y, iter, i, NN) = results_fifo.read();
    }
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
 * @summary: reduced port stream SpMV kernel
 */
// void krnl_spmv_multiport(const data_t *values, const data_t *col_index, const data_t *rowPtr, const data_t *row_length const data_t *x, data_t *y, const data_t NNZ, const data_t NN)
// {
// }
