#include "krnl_spmv.h"
#include <hls_stream.h>
#include <cstdio>

#define ARRAY {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512}

data_t LocalX[512] = ARRAY;

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
#pragma HLS interface mode = m_axi port = values bundle = gmem1
#pragma HLS interface mode = m_axi port = indices bundle = gmem0
#pragma HLS interface mode = m_axi port = x bundle = gmem
#pragma HLS interface mode = m_axi port = y bundle = gmem

    // batch_size iteration
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
   * @summary: reduced port stream SpMV kernel
   */
  void krnl_spmv_multi(const uintbuswidth_t *values, const uintbuswidth_t *indices,
                       const data_t *x, uintbuswidth_t *y, uint64_t batch_size)
  {
    // initialize the fifos and data stream
    // defines all the fifos
#pragma HLS interface mode = m_axi port = values bundle = gmem1
#pragma HLS interface mode = m_axi port = indices bundle = gmem0
//#pragma HLS interface mode = m_axi port = x bundle = gmem
#pragma HLS interface mode = m_axi port = y bundle = gmem1

    data_t x_tmp_0[512] = ARRAY;
    data_t x_tmp_1[512] = ARRAY;
    data_t x_tmp_2[512] = ARRAY;
    data_t x_tmp_3[512] = ARRAY;
    // batch_size iteration
    for (uint64_t iter = 0; iter < batch_size; iter++)
    {
//      cnt = 0;
//      // Load local X
//      for (index_t i = 0; i < MM+1; i++)
//      {
//#pragma HLS unroll factor = 16
//        data_t tmp = ARRAY2(x, iter, i, MM);
//        x_tmp_0[i] = tmp;
//        x_tmp_1[i] = tmp;
//        x_tmp_2[i] = tmp;
//        x_tmp_3[i] = tmp;
//      }
      krnl_spmv_multi_dataflow(iter, values, indices, x_tmp_0, x_tmp_1, x_tmp_2, x_tmp_3, y);
    }
  }

  void krnl_spmv_multi_dataflow(uint64_t iter, const uintbuswidth_t *values, const uintbuswidth_t *indices,
		  data_t *x_tmp_0, data_t *x_tmp_1, data_t *x_tmp_2, data_t *x_tmp_3, uintbuswidth_t *y)
  {
// #pragma HLS DATAFLOW
    const data_t indices_fifo_depth = 2;
    const data_t x_fifo_depth = 2;
    const data_t rows_fifo_depth = 2;
    const data_t values_fifo_depth = 2;
    const data_t cols_fifo_depth = 2;
    const data_t results_fifo_depth = 2;

    data_t indices_fifo_0[(NNZ + NN) / MULTI_FACTOR],
        rows_fifo_0[NN / MULTI_FACTOR],
        values_fifo_0[NNZ / MULTI_FACTOR],
        cols_fifo_0[NNZ / MULTI_FACTOR],
        results_fifo_0[NN / MULTI_FACTOR];
// #pragma HLS STREAM variable = indices_fifo_0 depth = indices_fifo_depth type = fifo
// #pragma HLS STREAM variable = rows_fifo_0 depth = rows_fifo_depth type = fifo
// #pragma HLS STREAM variable = values_fifo_0 depth = values_fifo_depth type = fifo
// #pragma HLS STREAM variable = cols_fifo_0 depth = cols_fifo_depth type = fifo
// #pragma HLS STREAM variable = results_fifo_0 depth = results_fifo_depth type = fifo

    data_t indices_fifo_1[(NNZ + NN) / MULTI_FACTOR],
        rows_fifo_1[NN / MULTI_FACTOR],
        values_fifo_1[NNZ / MULTI_FACTOR],
        cols_fifo_1[NNZ / MULTI_FACTOR],
        results_fifo_1[NN / MULTI_FACTOR];
// #pragma HLS STREAM variable = indices_fifo_1 depth = indices_fifo_depth type = fifo
// #pragma HLS STREAM variable = rows_fifo_1 depth = rows_fifo_depth type = fifo
// #pragma HLS STREAM variable = values_fifo_1 depth = values_fifo_depth type = fifo
// #pragma HLS STREAM variable = cols_fifo_1 depth = cols_fifo_depth type = fifo
// #pragma HLS STREAM variable = results_fifo_1 depth = results_fifo_depth type = fifo

    data_t indices_fifo_2[(NNZ + NN) / MULTI_FACTOR],
        rows_fifo_2[NN / MULTI_FACTOR],
        values_fifo_2[NNZ / MULTI_FACTOR],
        cols_fifo_2[NNZ / MULTI_FACTOR],
        results_fifo_2[NN / MULTI_FACTOR];
// #pragma HLS STREAM variable = indices_fifo_2 depth = indices_fifo_depth type = fifo
// #pragma HLS STREAM variable = rows_fifo_2 depth = rows_fifo_depth type = fifo
// #pragma HLS STREAM variable = values_fifo_2 depth = values_fifo_depth type = fifo
// #pragma HLS STREAM variable = cols_fifo_2 depth = cols_fifo_depth type = fifo
// #pragma HLS STREAM variable = results_fifo_2 depth = results_fifo_depth type = fifo

    data_t indices_fifo_3[(NNZ + NN) / MULTI_FACTOR],
        rows_fifo_3[NN / MULTI_FACTOR],
        values_fifo_3[NNZ / MULTI_FACTOR],
        cols_fifo_3[NNZ / MULTI_FACTOR],
        results_fifo_3[NN / MULTI_FACTOR];
// #pragma HLS STREAM variable = indices_fifo_3 depth = indices_fifo_depth type = fifo
// #pragma HLS STREAM variable = rows_fifo_3 depth = rows_fifo_depth type = fifo
// #pragma HLS STREAM variable = values_fifo_3 depth = values_fifo_depth type = fifo
// #pragma HLS STREAM variable = cols_fifo_3 depth = cols_fifo_depth type = fifo
// #pragma HLS STREAM variable = results_fifo_3 depth = results_fifo_depth type = fifo

    krnl_spmv_multi_load(iter, indices, indices_fifo_0, indices_fifo_1, indices_fifo_2, indices_fifo_3);
    krnl_spmv_multi_values(iter, values, values_fifo_0, values_fifo_1, values_fifo_2, values_fifo_3);

    krnl_spmv_multi_split(indices_fifo_0, rows_fifo_0, cols_fifo_0);
    krnl_spmv_multi_split(indices_fifo_1, rows_fifo_1, cols_fifo_1);
    krnl_spmv_multi_split(indices_fifo_2, rows_fifo_2, cols_fifo_2);
    krnl_spmv_multi_split(indices_fifo_3, rows_fifo_3, cols_fifo_3);
    krnl_spmv_multi_MAC(iter, x_tmp_0, rows_fifo_0, cols_fifo_0, values_fifo_0, results_fifo_0);
    krnl_spmv_multi_MAC(iter, x_tmp_1, rows_fifo_1, cols_fifo_1, values_fifo_1, results_fifo_1);
    krnl_spmv_multi_MAC(iter, x_tmp_2, rows_fifo_2, cols_fifo_2, values_fifo_2, results_fifo_2);
    krnl_spmv_multi_MAC(iter, x_tmp_3, rows_fifo_3, cols_fifo_3, values_fifo_3, results_fifo_3);

    krnl_spmv_multi_write(iter, y, results_fifo_0, results_fifo_1, results_fifo_2, results_fifo_3);
  }

void krnl_spmv_multi_load(uint64_t iter, const uintbuswidth_t *indices, data_t *indices_fifo_0, data_t *indices_fifo_1, data_t *indices_fifo_2, data_t *indices_fifo_3)
{
#pragma HLS inline off
  for (index_t i = 0; i < (NNZ + NN) / MULTI_FACTOR; i++)
  {
#pragma HLS pipeline
    uintbuswidth_t idx = ARRAY2(indices, iter, i, (NNZ + NN) / MULTI_FACTOR);
    indices_fifo_0[i] = (data_t)idx;
    indices_fifo_1[i] = (data_t)(idx >> DATA_WIDTH);
    indices_fifo_2[i] = (data_t)(idx >> (DATA_WIDTH * 2));
    indices_fifo_3[i] = (data_t)(idx >> (DATA_WIDTH * 3));
  }
}

void krnl_spmv_multi_values(uint64_t iter, const uintbuswidth_t *values, data_t *values_fifo_0, data_t *values_fifo_1, data_t *values_fifo_2, data_t *values_fifo_3)
{
#pragma HLS inline off
  for (index_t i = 0; i < NNZ / MULTI_FACTOR; i++)
  {
#pragma HLS pipeline
    uintbuswidth_t value = ARRAY2(values, iter, i, NNZ / MULTI_FACTOR);
    values_fifo_0[i] = (data_t)value;
    values_fifo_1[i] = (data_t)(value >> DATA_WIDTH);
    values_fifo_2[i] = (data_t)(value >> (DATA_WIDTH * 2));
    values_fifo_3[i] = (data_t)(value >> (DATA_WIDTH * 3));
  }
}

void krnl_spmv_multi_split(data_t *indices_fifo, data_t *rows_fifo, data_t *cols_fifo)
{
#pragma HLS inline off
  data_t col_left = 0;
  data_t rows_idx = 0;
  data_t cols_idx = 0;
  // feed data into the fifos
  // feed row index into fifo
  for (index_t i = 0; i < (NNZ + NN) / MULTI_FACTOR; i++)
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

void krnl_spmv_multi_MAC(uint64_t iter, data_t *x_tmp, data_t *rows_fifo, data_t *cols_fifo, data_t *values_fifo, data_t *results_fifo)
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
  data_t x;

  for (index_t i = 0; i < (NNZ + NN) / MULTI_FACTOR; i++)
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
      x = x_tmp[col];
      // printf("--- X ---: %d\n", x);
      cols_idx++;
      accumulator += value * x;

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

void krnl_spmv_multi_write(uint64_t iter, uintbuswidth_t *y, data_t *results_fifo_0, data_t *results_fifo_1, data_t *results_fifo_2, data_t *results_fifo_3)
{
#pragma HLS inline off
  // write back the accumulation to y vector
  for (index_t i = 0; i < NN / MULTI_FACTOR; i++)
  {
#pragma HLS pipeline
    uintbuswidth_t result = (uintbuswidth_t)results_fifo_3[i] << (DATA_WIDTH * 3) |
                            (uintbuswidth_t)results_fifo_2[i] << (DATA_WIDTH * 2) |
                            (uintbuswidth_t)results_fifo_1[i] << (DATA_WIDTH * 1) |
                            (uintbuswidth_t)results_fifo_0[i];

    ARRAY2(y, iter, i, NN / MULTI_FACTOR) = result;
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
