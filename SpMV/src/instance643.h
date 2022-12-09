#pragma once

#define DATA_WIDTH 32

#define MULTI_FACTOR 4

#define BUS_BIT_WIDTH 128

typedef ap_uint<BUS_BIT_WIDTH> uintbuswidth_t;

typedef unsigned int data_t;

static inline bool nearlyEqual(data_t a, data_t b) { return a == b; }

#define NNZ 512 // Non-zero elements

#define MM 512 // Size of vector X

#define NN 512 // Size of vector Y
