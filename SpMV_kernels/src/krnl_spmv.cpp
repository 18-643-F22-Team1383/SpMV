#include "krnl_spmv.h"
#include <ap_int.h>
#include <hls_stream.h>
#include <stdio.h>
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
                          const data_t *x, data_t *y, uint64_t batch_size){

    for (uint64_t iter = 0; iter < batch_size; iter++)
    {
    // initializa parameters and corresponding fifos
    const data_t indices_fifo_depth = NNZ + NN;
    const data_t rows_fifo_depth = NN;
    const data_t values_fifo_depth = NNZ;
    const data_t col_fifo_depth = NN;


    hls::stream<data_t> indices_fifo;
  #pragma HLS STREAM variable = indices_fifo depth = indices_fifo_depth type = fifo

    hls::stream<data_t> values_fifo;
  #pragma HLS STREAM variable = values_fifo depth = values_fifo_depth type = fifo

    hls::stream<data_t> row_fifo;
  #pragma HLS STREAM variable = row_fifo depth = rows_fifo_depth type = fifo

    hls::stream<data_t> results_fifo;
  #pragma HLS STREAM variable = results_fifo depth = rows_fifo_depth type = fifo

    hls::stream<data_t> col_fifo;
  #pragma HLS STREAM variable = col_fifo depth = col_fifo_depth type = fifo


  // feed indice fifo
    for (index_t i = 0; i < NNZ + NN; i++)
    {
      #pragma HLS pipeline
        indices_fifo <<  ARRAY2(indices, iter, i, NNZ + NN);
    }
  // feed values fifo
    data_t idx_col_left = 0;
    for (index_t i = 0; i < NNZ; i++)
    {
      #pragma HLS pipeline
        values_fifo << ARRAY2(values,iter,i,NNZ);
    }
  
  // read indice fifo assign colIndx or colLeft
    for (index_t i = 0; i < NNZ+NN; i++)
    {
      #pragma HLS pipeline
      data_t idx = indices_fifo.read();
      // last row is calculated or kernel just started 
      if(idx_col_left == 0){
        idx_col_left = idx;
        row_fifo << idx_col_left;
      }else{
      // current indice is the col index
        col_fifo << idx;
        idx_col_left -- ;
      }
    }
    
  // accumulate and reduce statge
  data_t accumulator = 0;
  data_t value = 0;
  data_t col;
  for (index_t i = 0; i < NNZ+NN; i++)
  {
    #pragma HLS pipeline
    // first computation or row compute finished
    if(i == 0 || idx_col_left == 0){
      idx_col_left = row_fifo.read();
      accumulator =0;
    }else{
      for(index_t i = 0; i < idx_col_left; i ++){
      #pragma HLS pipeline
          value = values_fifo.read();
          col = col_fifo.read();
          accumulator += value * ARRAY2(x, iter, col, MM);
      }
      results_fifo << accumulator;
    }
  }

// write back to the Y vector
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
 * @summary: fast stream SpMV kernel
 */
  void krnl_spmv_fast(const data_t *values, const data_t *col_index, const data_t *rowPtr,
                        const data_t *x, data_t *y, uint64_t batch_size = 10)
    {

  // initialize the fifos and data stream
  #pragma HLS DATAFLOW
      const data_t row_length_fifo_depth = NN;
      const data_t values_fifo_depth = NN;
      const data_t cols_fifo_depth = NN;
      const data_t results_fifo_depth = NN;
      // defines all the fifos
      hls::stream<data_t> row_length_fifo;
  #pragma HLS STREAM variable = row_length_fifo depth = row_length_fifo_depth type = fifo

      hls::stream<data_t> values_fifo;
  #pragma HLS STREAM variable = values_fifo depth = values_fifo_depth type = fifo

      hls::stream<data_t> cols_fifo;
  #pragma HLS STREAM variable = cols_fifo depth = cols_fifo_depth type = fifo

      hls::stream<data_t> results_fifo;
  #pragma HLS STREAM variable = results_fifo depth = results_fifo_depth type = fifo

      // batch_size iteration
  batch_loop:
      for (uint64_t iter = 0; iter < batch_size; iter++)
      {

        // initialize row_length vector
      data_t row_length[NN];

      row_length_initialization:
      for (index_t i = 0; i < NN; i++){
        #pragma HLS pipeline
            printf("generate row-length\n");
            row_length[i] = ARRAY2(rowPtr, iter, i + 1, NN + 1) - ARRAY2(rowPtr, iter, i, NN + 1);
        }

      row_length_fifo_initialization:
      for (index_t i = 0; i < NN; i++){
        #pragma HLS pipeline
          // printf("feed row length into fifo\n");
          row_length_fifo << row_length[i];
        }

        // feed column index and values into fifo
      cols_values_fifo_initialization:
        for (index_t i = 0; i < NNZ; i++)
            {
      #pragma HLS pipeline
            if(values_fifo.empty()){
            printf("feed values into fifo: values_fifo empty\n");
            }
            printf("feed values into fifo\n");
              values_fifo << ARRAY2(values, iter, i, NNZ);
            // printf("feed cols into fifo\n");
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
      compute_loop:
        for (index_t i = 0; i < NN; i++)
        {

  #pragma HLS pipeline
          // printf("read col_left from row-length\n");
          if(values_fifo.empty()){
            printf("compute_loop: values_fifo empty\n");
          }
          
          
            col_left = row_length_fifo.read();
            accumulator = 0;
        col_left_loop:
          for (index_t j = 0; j < col_left;j++)
          {
          printf("read value from values fifo\n"); // used to suck here
            value = values_fifo.read();
          // printf("read colIndx from cols fifo\n");
            col = cols_fifo.read();
            
            accumulator += value * ARRAY2(x,iter,col,MM);
          }
          printf("write accumulator into result fifo\n");
          results_fifo << accumulator;
        }
        // write back the accumulation to y vector
      write_back_loop:
        for (index_t j = 0; j < NN; j++)
        {
    #pragma HLS pipeline
          printf("read result result fifo\n");
          data_t result = results_fifo.read();
      //   printf("batch %d j %d current result %d\n",iter,j,result);    
          ARRAY2(y, iter, j, NN) = result;
        }
      }
    }

#ifdef __VITIS_CL__ // for lab 3
} // extern
#endif
